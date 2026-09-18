import pytest

from assay.corpus import SealedCorpus, index_corpus, split_and_seal
from assay.errors import SealError


def test_index_hashes_every_document(manifest):
    assert len(manifest.documents) == 12
    assert all(len(d.sha256) == 64 for d in manifest.documents)
    assert manifest.duplicates() == []


def test_corpus_spans_an_order_of_magnitude(manifest):
    lo, hi = manifest.size_span()
    assert hi / lo > 10, "the context ladder needs a real size range to fit a slope"


def test_manifest_round_trips(manifest, tmp_path):
    from assay.corpus import CorpusManifest

    path = tmp_path / "corpus-manifest.json"
    manifest.save(path)
    again = CorpusManifest.load(path)
    assert again.hash() == manifest.hash()


def test_split_is_by_document_and_disjoint(sealed):
    fit = set(sealed.seal.fit)
    blind = set(sealed.seal.blind)
    assert fit and blind
    assert fit.isdisjoint(blind)
    assert fit | blind == set(sealed.manifest.names)


def test_split_is_deterministic_under_a_seed(manifest):
    a = split_and_seal(manifest, seed=99)
    b = split_and_seal(manifest, seed=99)
    assert a.seal_sha256 == b.seal_sha256
    assert split_and_seal(manifest, seed=100).seal_sha256 != a.seal_sha256


def test_reading_a_blind_document_raises(sealed):
    blind_name = sealed.seal.blind[0]
    with pytest.raises(SealError, match="blind split"):
        sealed.read(blind_name)


def test_reading_a_blind_document_works_when_intent_is_declared(sealed):
    blind_name = sealed.seal.blind[0]
    assert sealed.read(blind_name, allow_blind=True)


def test_listing_the_blind_split_requires_intent(sealed):
    with pytest.raises(SealError):
        sealed.blind_documents()
    assert sealed.blind_documents(allow_blind=True)


def test_fit_documents_never_include_blind_material(sealed):
    sealed.assert_fit_only([d.name for d in sealed.fit_documents()])


def test_assert_fit_only_catches_a_leak(sealed):
    with pytest.raises(SealError, match="reached a fit-split operation"):
        sealed.assert_fit_only([sealed.seal.blind[0]])


def test_editing_a_saved_seal_is_detected(sealed, tmp_path):
    import json

    from assay.corpus import Seal

    path = tmp_path / "seal.json"
    sealed.seal.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    moved = data["blind"].pop()
    data["fit"].append(moved)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SealError, match="edited since it was sealed"):
        Seal.load(path)


def test_a_seal_from_a_different_corpus_is_refused(manifest, corpus_dir, tmp_path):
    (tmp_path / "extra.txt").write_text("x" * 500, encoding="utf-8")
    for p in corpus_dir.glob("*.txt"):
        (tmp_path / p.name).write_bytes(p.read_bytes())
    other = index_corpus(tmp_path)
    seal = split_and_seal(other, seed=1)
    with pytest.raises(SealError, match="different corpus manifest"):
        SealedCorpus(manifest, seal)


def test_bands_are_ordered_by_size(sealed):
    bands = sealed.bands([d.name for d in sealed.fit_documents()], n_bands=3)
    medians = [sealed.size(b[len(b) // 2]) for b in bands if b]
    assert medians == sorted(medians)


def test_too_small_a_corpus_cannot_be_split(tmp_path):
    for i in range(3):
        (tmp_path / f"d{i}.txt").write_text("x" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="at least 4 documents"):
        split_and_seal(index_corpus(tmp_path))
