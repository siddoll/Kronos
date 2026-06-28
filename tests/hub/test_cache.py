from hub.data.cache import OHLCVCache

def test_put_then_get_roundtrips(tmp_path, make_df):
    cache = OHLCVCache(str(tmp_path))
    df = make_df([10, 11, 12])
    assert cache.get("AAA") is None
    cache.put("AAA", df)
    out = cache.get("AAA")
    assert out is not None and list(out["close"]) == [10, 11, 12]

def test_get_returns_none_when_expired(tmp_path, make_df):
    cache = OHLCVCache(str(tmp_path), ttl_hours=0)
    cache.put("AAA", make_df([1, 2, 3]))
    assert cache.get("AAA") is None
