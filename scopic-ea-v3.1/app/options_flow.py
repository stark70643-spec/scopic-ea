"""Options-derived analytics via Tradier (sandbox or production).

Sandbox (developer.tradier.com, free, no brokerage needed) provides delayed
quotes + option chains WITH GREEKS (delta/gamma/theta/vega/mid_iv) computed
server-side, so we no longer need to compute Black-Scholes ourselves for
gamma - we just aggregate what Tradier returns. Vanna still isn't provided
by any vendor at this tier, so we compute that one greek in-house from
Tradier's IV using Black-Scholes.

Proxies: GLD for GC, QQQ for NQ (same rationale as before - CME futures
options aren't available on Tradier's retail API).

IMPORTANT (also told to the AI): this is positioning CONTEXT from a delayed
feed and once-daily OI settles. Never an entry trigger.
"""
import math
import time
from datetime import datetime, timezone

import httpx

from . import config

PROXY = {"GC": "GLD", "NQ": "QQQ"}
CACHE_SEC = 600
RISK_FREE = 0.04

_cache = {}  # instrument -> (ts, summary)


def _headers():
    return {"Authorization": f"Bearer {config.TRADIER_API_KEY}", "Accept": "application/json"}


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _d1(S, K, T, r, iv):
    return (math.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))


def _vanna(S, K, T, r, iv):
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = _d1(S, K, T, r, iv)
        d2 = d1 - iv * math.sqrt(T)
        return -_norm_pdf(d1) * d2 / iv
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


async def _get(client, path, **params):
    r = await client.get(f"{config.TRADIER_BASE_URL}{path}", params=params, headers=_headers())
    r.raise_for_status()
    return r.json()


async def compute(instrument: str, futures_price: float | None = None) -> dict:
    proxy = PROXY.get(instrument)
    if not proxy:
        return {"error": "unsupported instrument"}
    if not config.TRADIER_API_KEY:
        return {"error": "TRADIER_API_KEY not configured on the backend"}

    cached = _cache.get(instrument)
    if cached and time.time() - cached[0] < CACHE_SEC:
        out = dict(cached[1])
        out["cache_age_sec"] = int(time.time() - cached[0])
        if futures_price:
            _translate(out, futures_price)
        return out

    try:
        async with httpx.AsyncClient(timeout=20) as c:
            qj = await _get(c, "/markets/quotes", symbols=proxy)
            q = qj.get("quotes", {}).get("quote")
            if isinstance(q, list):
                q = q[0]
            spot = q.get("last") or q.get("close")
            if not spot:
                return {"error": f"no quote for {proxy}"}

            ej = await _get(c, "/markets/options/expirations", symbol=proxy, includeAllRoots="true")
            dates = ej.get("expirations", {}).get("date") or []
            if isinstance(dates, str):
                dates = [dates]
            now = time.time()
            dates = sorted(dates)[:4]  # nearest 4 expiries

            per_strike = {}
            term = []
            slice0 = {"net_gex": 0.0, "call_oi": 0, "put_oi": 0, "dte_days": None}
            skew = None

            for i, exp in enumerate(dates):
                cj = await _get(c, "/markets/options/chains", symbol=proxy, expiration=exp, greeks="true")
                rows = cj.get("options", {}).get("option") or []
                if isinstance(rows, dict):
                    rows = [rows]
                exp_ts = datetime.strptime(exp, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc).timestamp()
                T = max((exp_ts - now) / 86400.0, 0.02) / 365.0
                dte = round((exp_ts - now) / 86400.0, 1)

                atm_iv, atm_dist = None, 1e9
                iv_put_lo, iv_call_hi = None, None

                for o in rows:
                    K = o.get("strike")
                    oi = o.get("open_interest") or 0
                    kind = "call" if o.get("option_type") == "call" else "put"
                    greeks = o.get("greeks") or {}
                    gamma = greeks.get("gamma")
                    iv = greeks.get("mid_iv") or greeks.get("smv_vol")
                    if not K or gamma is None or not iv or iv <= 0.005 or iv > 5:
                        continue

                    gex = gamma * oi * 100 * spot * spot * 0.01
                    vn = _vanna(spot, K, T, RISK_FREE, iv) * oi * 100 * spot * 0.01
                    sgn = 1 if kind == "call" else -1

                    s = per_strike.setdefault(K, {"gex": 0.0, "vanna": 0.0,
                                                  "call_oi": 0, "put_oi": 0,
                                                  "call_gex": 0.0, "put_gex": 0.0})
                    s["gex"] += sgn * gex
                    s["vanna"] += sgn * vn
                    s[kind + "_oi"] += oi
                    s[kind + "_gex"] += gex
                    if i == 0:
                        slice0["net_gex"] += sgn * gex
                        slice0[kind + "_oi"] += oi
                        slice0["dte_days"] = dte

                    d = abs(K - spot)
                    if d < atm_dist:
                        atm_dist, atm_iv = d, iv
                    m = K / spot
                    if kind == "put" and 0.93 <= m <= 0.97:
                        iv_put_lo = iv if iv_put_lo is None else (iv_put_lo + iv) / 2
                    if kind == "call" and 1.03 <= m <= 1.07:
                        iv_call_hi = iv if iv_call_hi is None else (iv_call_hi + iv) / 2

                if atm_iv:
                    term.append({"dte_days": dte, "atm_iv_pct": round(atm_iv * 100, 1)})
                if i == 0 and iv_put_lo and iv_call_hi:
                    skew = round((iv_put_lo - iv_call_hi) * 100, 1)

            strikes = sorted(per_strike)
            if not strikes:
                return {"error": f"no usable option/greek data for {proxy} "
                                  "(sandbox feed may lack greeks for this symbol)"}

            net_gex = sum(v["gex"] for v in per_strike.values())
            call_wall = max(strikes, key=lambda k: per_strike[k]["call_gex"], default=None)
            put_wall = max(strikes, key=lambda k: per_strike[k]["put_gex"], default=None)
            oi_top = sorted(strikes, key=lambda k: per_strike[k]["call_oi"] + per_strike[k]["put_oi"],
                            reverse=True)[:5]

            flip = None
            cum, prev_k, prev_cum = 0.0, None, 0.0
            for k in strikes:
                cum += per_strike[k]["gex"]
                if prev_k is not None and prev_cum < 0 <= cum:
                    flip = round((prev_k + k) / 2, 2)
                prev_k, prev_cum = k, cum

            vanna_net = sum(v["vanna"] for v in per_strike.values())
            profile = [{"strike": k, "net_gex_musd": round(per_strike[k]["gex"] / 1e6, 1)}
                       for k in strikes if abs(per_strike[k]["gex"]) > 1e5][:60]

            summary = {
                "instrument": instrument, "proxy": proxy, "proxy_spot": spot,
                "source": "tradier", "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                "quote_delay_note": "Tradier sandbox quotes are delayed; OI is prior-day settle",
                "net_gex_musd": round(net_gex / 1e6, 1),
                "dealer_positioning": "SHORT_GAMMA (moves amplify)" if net_gex < 0
                                       else "LONG_GAMMA (moves dampen/pin)",
                "zero_gamma_flip_proxy": flip,
                "call_wall_proxy": call_wall, "put_wall_proxy": put_wall,
                "top_oi_strikes_proxy": oi_top,
                "net_vanna": round(vanna_net / 1e6, 2),
                "vanna_read": "vol-down supports price" if vanna_net > 0 else "vol-down pressures price",
                "iv_term_structure": term,
                "put_call_skew_pts": skew,
                "zero_dte_slice": {**slice0, "net_gex_musd": round(slice0["net_gex"] / 1e6, 1),
                                   "afternoon_note": "intraday 0DTE flow invisible in free data; weight low late-session"},
                "gex_profile": profile,
            }
            summary["zero_dte_slice"].pop("net_gex", None)
            _cache[instrument] = (time.time(), summary)
            out = dict(summary)
            out["cache_age_sec"] = 0
            if futures_price:
                _translate(out, futures_price)
            return out
    except httpx.HTTPStatusError as e:
        msg = f"Tradier {e.response.status_code}: {e.response.text[:200]}"
        if cached:
            out = dict(cached[1]); out["cache_age_sec"] = int(time.time() - cached[0]); out["stale_error"] = msg
            return out
        return {"instrument": instrument, "error": msg}
    except Exception as e:
        if cached:
            out = dict(cached[1]); out["cache_age_sec"] = int(time.time() - cached[0]); out["stale_error"] = str(e)
            return out
        return {"instrument": instrument, "error": f"options fetch failed: {e}"}


def _translate(out: dict, fut_price: float):
    spot = out.get("proxy_spot")
    if not spot or not fut_price:
        return
    ratio = fut_price / spot
    out["futures_price_used"] = fut_price
    for k_src, k_dst in (("zero_gamma_flip_proxy", "zero_gamma_flip_futures"),
                         ("call_wall_proxy", "call_wall_futures"),
                         ("put_wall_proxy", "put_wall_futures")):
        v = out.get(k_src)
        out[k_dst] = round(v * ratio, 1) if v else None


def compact_for_ai(summary: dict) -> dict:
    if not summary or summary.get("error"):
        return {"unavailable": summary.get("error", "no data")}
    keys = ["proxy", "source", "as_of_utc", "cache_age_sec", "quote_delay_note", "net_gex_musd",
            "dealer_positioning", "zero_gamma_flip_proxy", "zero_gamma_flip_futures",
            "call_wall_proxy", "call_wall_futures", "put_wall_proxy", "put_wall_futures",
            "top_oi_strikes_proxy", "net_vanna", "vanna_read", "iv_term_structure",
            "put_call_skew_pts", "zero_dte_slice"]
    return {k: summary.get(k) for k in keys if summary.get(k) is not None}
