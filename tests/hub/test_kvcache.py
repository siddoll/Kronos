from hub.data.kvcache import KVCache

def test_put_get_roundtrip(tmp_path):
    kv = KVCache(str(tmp_path))
    assert kv.get("AAPL") is None
    kv.put("AAPL", {"pe_ratio": 30.0})
    assert kv.get("AAPL") == {"pe_ratio": 30.0}

def test_ttl_expiry(tmp_path):
    kv = KVCache(str(tmp_path), ttl_hours=0)
    kv.put("X", [1, 2])
    assert kv.get("X") is None

def test_key_with_slash(tmp_path):
    kv = KVCache(str(tmp_path))
    kv.put("BRK/B", {"a": 1})
    assert kv.get("BRK/B") == {"a": 1}
