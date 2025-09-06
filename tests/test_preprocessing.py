from src.preprocessing import clean_text

def test_clean_text_basic():
    t = "இது ஒரு டெஸ்ட்! http://example.com @user"
    out = clean_text(t)
    assert "http" not in out and "@user" not in out
    assert isinstance(out, str)
