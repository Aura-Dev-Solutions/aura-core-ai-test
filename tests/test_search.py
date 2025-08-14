from app.services.index import FaissIndex

def test_search_basic():
    idx = FaissIndex()
    idx.add("1", "this is an invoice with amount $12345 and due date 2024-10-22")
    idx.add("2", "this contract has a signature and a clause about term limits")
    idx.build()
    res = idx.search("invoice", top_k=1)
    assert res and res[0]["doc_id"] == "1"
