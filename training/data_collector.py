"""
Anad Training Data Collector
=============================
Collects clean, consented, public domain training data.

Sources (all free, all legal, all ethical):
  1. Project Gutenberg   — 70,000 free books
  2. Wikipedia           — all languages
  3. ArXiv abstracts     — scientific papers
  4. Common Crawl        — filtered web text
  5. Stack Exchange      — Q&A, CC licensed
  6. OpenSubtitles       — dialogue, conversations
  7. Indic corpus        — Indian language texts

Every source documented. Nothing hidden.
No private data. No scraped social media.
No copyrighted content without permission.

Author: Anad Community
License: Public Domain
"""

import os
import json
import hashlib
import time
import urllib.request
import urllib.error
from typing import List, Dict, Iterator, Optional
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════
# DATA RECORD — every piece of training data tracked
# ══════════════════════════════════════════════════════════════════

@dataclass
class DataRecord:
    """
    A single training example with full provenance.
    We know exactly where every piece of data came from.
    """
    text: str
    source: str          # gutenberg / wikipedia / arxiv / etc
    language: str        # en / gu / hi / ta / etc
    license: str         # public_domain / cc0 / cc_by / etc
    url: str = ""
    title: str = ""
    checksum: str = ""   # sha256 of text — detect duplicates

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.sha256(
                self.text.encode()
            ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "language": self.language,
            "license": self.license,
            "url": self.url,
            "title": self.title,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataRecord":
        return cls(**d)


# ══════════════════════════════════════════════════════════════════
# DATASET INDEX — tracks what data exists and where
# ══════════════════════════════════════════════════════════════════

class DatasetIndex:
    """
    Tracks all training data.
    Prevents duplicate training.
    Enables sharing between nodes.

    Key insight:
      Each piece of data has a checksum.
      Before training on anything, check if already trained.
      Share the index with peers so they skip what you did.
    """

    def __init__(self, index_path: str):
        self.index_path = index_path
        self._seen: set = set()          # checksums of seen data
        self._records: List[dict] = []   # all records metadata
        self._load()

    def is_seen(self, checksum: str) -> bool:
        return checksum in self._seen

    def mark_seen(self, record: DataRecord):
        self._seen.add(record.checksum)
        self._records.append({
            "checksum": record.checksum,
            "source": record.source,
            "language": record.language,
            "title": record.title[:80],
            "timestamp": time.time(),
        })
        if len(self._records) % 100 == 0:
            self._save()

    def total_seen(self) -> int:
        return len(self._seen)

    def stats(self) -> dict:
        sources = {}
        languages = {}
        for r in self._records:
            sources[r["source"]] = sources.get(r["source"], 0) + 1
            languages[r["language"]] = languages.get(r["language"], 0) + 1
        return {
            "total": len(self._seen),
            "by_source": sources,
            "by_language": languages,
        }

    def export_seen_checksums(self) -> List[str]:
        """Share with peers so they skip what we already trained on"""
        return list(self._seen)

    def import_seen_checksums(self, checksums: List[str]):
        """Import from peer — skip their already-trained data"""
        before = len(self._seen)
        self._seen.update(checksums)
        added = len(self._seen) - before
        if added > 0:
            print(f"  Imported {added} seen checksums from peer")
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        data = {
            "seen": list(self._seen),
            "records": self._records[-10000:],  # keep last 10k
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load(self):
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, encoding="utf-8") as f:
                data = json.load(f)
            self._seen = set(data.get("seen", []))
            self._records = data.get("records", [])
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# DATA SOURCES
# ══════════════════════════════════════════════════════════════════

class GutenbergSource:
    """
    Project Gutenberg — 70,000 free books.
    All public domain. All legal.
    Best source for clean English text.
    """

    # Curated list of high-quality books
    BOOK_IDS = [
        # Philosophy & wisdom
        (2680, "Meditations — Marcus Aurelius", "en"),
        (1232, "The Prince — Machiavelli", "en"),
        (4280, "Pragmatism — William James", "en"),
        # Science
        (1228, "The Origin of Species — Darwin", "en"),
        (5001, "Relativity — Einstein", "en"),
        # Literature
        (1342, "Pride and Prejudice — Austen", "en"),
        (2701, "Moby Dick — Melville", "en"),
        (84,   "Frankenstein — Shelley", "en"),
        (11,   "Alice in Wonderland — Carroll", "en"),
        # History
        (2600, "War and Peace — Tolstoy", "en"),
        (98,   "A Tale of Two Cities — Dickens", "en"),
        # Indian classics (English translations)
        (16955,"The Mahabharata — translated", "en"),
        (7864, "The Ramayana — translated", "en"),
        (3283, "Kamasutra — Vatsyayana", "en"),
    ]

    BASE_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
    ALT_URL  = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

    def fetch(self, book_id: int, title: str, language: str) -> Optional[DataRecord]:
        """Fetch a single book from Gutenberg"""
        for url_template in [self.BASE_URL, self.ALT_URL]:
            url = url_template.format(id=book_id)
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Anad-AI/0.1 (public domain training)"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")

                # Strip Gutenberg header/footer
                text = self._clean(raw)
                if len(text) < 1000:
                    continue

                return DataRecord(
                    text=text,
                    source="gutenberg",
                    language=language,
                    license="public_domain",
                    url=url,
                    title=title,
                )
            except Exception:
                continue
        return None

    def _clean(self, text: str) -> str:
        """Remove Gutenberg boilerplate"""
        lines = text.split("\n")
        start, end = 0, len(lines)
        for i, line in enumerate(lines):
            if "*** START OF" in line or "***START OF" in line:
                start = i + 1
                break
        for i, line in enumerate(lines):
            if "*** END OF" in line or "***END OF" in line:
                end = i
                break
        return "\n".join(lines[start:end]).strip()

    def stream(self) -> Iterator[DataRecord]:
        """Yield books one by one"""
        for book_id, title, language in self.BOOK_IDS:
            print(f"  Fetching: {title}...")
            record = self.fetch(book_id, title, language)
            if record:
                yield record
                time.sleep(1)  # be polite to Gutenberg servers


class WikipediaSource:
    """
    Wikipedia — clean, factual, multilingual.
    CC-BY-SA licensed — free for training.
    Prioritize Indian language articles.
    """

    # Curated article titles by language
    ARTICLES = {
        "en": [
            "Artificial intelligence", "Machine learning", "Neural network",
            "India", "Gujarat", "Sanskrit", "History of mathematics",
            "Philosophy of mind", "Computer science", "Open source",
            "Internet", "Democracy", "Human rights", "Climate change",
            "Quantum mechanics", "Evolution", "DNA", "Cell biology",
        ],
        "hi": [
            "कृत्रिम बुद्धिमत्ता", "भारत", "हिन्दी", "विज्ञान",
            "गणित", "दर्शनशास्त्र", "इतिहास",
        ],
        "gu": [
            "ગુજરાત", "ભારત", "ગુજરાતી ભાષા", "કૃત્રિમ બુદ્ધિ",
            "વિજ્ઞાન", "ગણિત",
        ],
        "ta": [
            "செயற்கை நுண்ணறிவு", "இந்தியா", "தமிழ்நாடு",
        ],
    }

    API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    API_URL_LANG = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

    def fetch_article(self, title: str, lang: str = "en") -> Optional[DataRecord]:
        """Fetch a Wikipedia article summary"""
        try:
            if lang == "en":
                url = self.API_URL.format(title=title.replace(" ", "_"))
            else:
                url = self.API_URL_LANG.format(
                    lang=lang,
                    title=title.replace(" ", "_")
                )
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Anad-AI/0.1 (public domain training)",
                    "Accept": "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            text = data.get("extract", "")
            if len(text) < 100:
                return None

            return DataRecord(
                text=text,
                source="wikipedia",
                language=lang,
                license="cc_by_sa",
                url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                title=data.get("title", title),
            )
        except Exception:
            return None

    def stream(self) -> Iterator[DataRecord]:
        """Yield articles across all languages"""
        for lang, titles in self.ARTICLES.items():
            for title in titles:
                record = self.fetch_article(title, lang)
                if record:
                    print(f"  Wikipedia [{lang}]: {title}")
                    yield record
                    time.sleep(0.5)


class IndicCorpusSource:
    """
    Handcrafted Indian language training data.
    Covers Gujarati, Hindi, Tamil, Telugu, Bengali.

    This is where Anad differs from every other model —
    Indian languages are first class, not afterthoughts.
    """

    # Seed texts in Indian languages
    # More will be added by community contributions
    SEED_TEXTS = [
        # Gujarati
        DataRecord(
            text="""ગુજરાત ભારતનું એક રાજ્ય છે. ગુજરાતી ભાષા ઈન્ડો-આર્યન ભાષા પરિવારની છે.
ગાંધીજીનો જન્મ ગુજરાતના પોરબંદરમાં થયો હતો. ગુજરાત તેના ઉદ્યોગ અને વ્યાપાર માટે પ્રખ્યાત છે.
ભારતની આઝાદીમાં ગુજરાતીઓએ મહત્વની ભૂમિકા ભજવી હતી.""",
            source="indic_seed", language="gu",
            license="public_domain", title="Gujarat intro"
        ),
        DataRecord(
            text="""અનાદ એ સૌ માટે AI છે. કોઈ કોર્પોરેશન નહીં, કોઈ સરકાર નહીં.
ફક્ત લોકો, ફક્ત જ્ઞાન, ફક્ત સ્વતંત્રતા.
તમારો ડેટા તમારો છે. તમારી યાદ તમારી છે. AI બધા માટે છે.""",
            source="indic_seed", language="gu",
            license="public_domain", title="Anad mission Gujarati"
        ),
        # Hindi
        DataRecord(
            text="""भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं।
हिन्दी भारत की राजभाषा है। भारत में अनेक संस्कृतियाँ एक साथ रहती हैं।
विज्ञान और तकनीक के क्षेत्र में भारत आगे बढ़ रहा है।""",
            source="indic_seed", language="hi",
            license="public_domain", title="India Hindi intro"
        ),
        DataRecord(
            text="""कृत्रिम बुद्धिमत्ता एक ऐसी तकनीक है जो मशीनों को सोचने की क्षमता देती है।
अनाद एक ऐसा AI है जो सबका है, किसी कंपनी का नहीं।
आपका डेटा आपका है। आपकी याददाश्त आपकी है।""",
            source="indic_seed", language="hi",
            license="public_domain", title="AI Hindi intro"
        ),
        # Tamil
        DataRecord(
            text="""தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்று.
இந்தியா பல மொழிகள் மற்றும் கலாச்சாரங்களின் தாயகம்.
அனாத் என்பது அனைவருக்கும் சொந்தமான AI.""",
            source="indic_seed", language="ta",
            license="public_domain", title="Tamil intro"
        ),
        # Telugu
        DataRecord(
            text="""తెలుగు భారతదేశంలో అత్యధికంగా మాట్లాడే భాషలలో ఒకటి.
అనాద్ అనేది అందరి కోసం AI. ఏ కంపెనీకీ చెందినది కాదు.""",
            source="indic_seed", language="te",
            license="public_domain", title="Telugu intro"
        ),
        # Sanskrit
        DataRecord(
            text="""अनादि अनन्तं ब्रह्म। सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।
सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्।
विद्या ददाति विनयं विनयाद्याति पात्रताम्।""",
            source="indic_seed", language="sa",
            license="public_domain", title="Sanskrit shlokas"
        ),
        # English — conversational
        DataRecord(
            text="""Hello, how can I help you today?
I can assist with questions, writing, coding, and analysis.
Feel free to ask me anything. I will do my best to help.""",
            source="indic_seed", language="en",
            license="public_domain", title="Conversation seed"
        ),
        DataRecord(
            text="""Anad is public AI that belongs to everyone.
No corporation owns it. No government controls it.
Your data stays on your device. Your memory is yours.
The network grows stronger with every node that joins.""",
            source="indic_seed", language="en",
            license="public_domain", title="Anad description"
        ),
    ]

    def stream(self) -> Iterator[DataRecord]:
        for record in self.SEED_TEXTS:
            yield record


# ══════════════════════════════════════════════════════════════════
# MAIN DATA COLLECTOR
# ══════════════════════════════════════════════════════════════════

class AnadDataCollector:
    """
    Orchestrates all data collection.

    Tracks what has been seen.
    Skips duplicates.
    Saves to disk in chunks.
    Shareable index for peer nodes.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.index = DatasetIndex(
            os.path.join(data_dir, "index.json")
        )
        self._chunk_size = 1000
        self._current_chunk: List[dict] = []
        self._chunk_num = 0

    def collect_all(
        self,
        include_gutenberg: bool = True,
        include_wikipedia: bool = True,
        include_indic: bool = True,
        max_records: int = 50000,
    ):
        """
        Collect training data from all sources.
        Skips anything already seen.
        """
        print("\n" + "═" * 50)
        print("  ANAD DATA COLLECTION")
        print(f"  Already seen: {self.index.total_seen()} records")
        print("═" * 50 + "\n")

        collected = 0

        # Always include Indic seed data first
        if include_indic:
            print("Collecting Indic seed data...")
            for record in IndicCorpusSource().stream():
                if self._add(record):
                    collected += 1

        # Wikipedia — multilingual
        if include_wikipedia and collected < max_records:
            print("\nCollecting Wikipedia articles...")
            for record in WikipediaSource().stream():
                if self._add(record):
                    collected += 1
                if collected >= max_records:
                    break

        # Gutenberg — books
        if include_gutenberg and collected < max_records:
            print("\nCollecting Project Gutenberg books...")
            for record in GutenbergSource().stream():
                if self._add(record):
                    collected += 1
                if collected >= max_records:
                    break

        # Save final chunk
        self._flush()

        print("\n" + "═" * 50)
        print(f"  Collection complete")
        print(f"  New records: {collected}")
        print(f"  Total seen:  {self.index.total_seen()}")
        stats = self.index.stats()
        print(f"  By language: {stats['by_language']}")
        print("═" * 50 + "\n")

    def _add(self, record: DataRecord) -> bool:
        """Add a record if not seen before"""
        if self.index.is_seen(record.checksum):
            return False

        self.index.mark_seen(record)
        self._current_chunk.append(record.to_dict())

        if len(self._current_chunk) >= self._chunk_size:
            self._flush()

        return True

    def flush(self):
        """Public flush — call after adding records"""
        self._flush()

    def _flush(self):
        """Save current chunk to disk"""
        if not self._current_chunk:
            return

        chunk_path = os.path.join(
            self.data_dir,
            f"chunk_{self._chunk_num:04d}.jsonl"
        )
        with open(chunk_path, "w", encoding="utf-8") as f:
            for record in self._current_chunk:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  Saved chunk {self._chunk_num} ({len(self._current_chunk)} records)")
        self._chunk_num += 1
        self._current_chunk = []

    def stream_for_training(self) -> Iterator[str]:
        """
        Stream training texts from saved chunks.
        Used by the trainer.
        """
        chunk_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.endswith(".jsonl")
        ])

        for chunk_file in chunk_files:
            path = os.path.join(self.data_dir, chunk_file)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        yield record["text"]
                    except Exception:
                        continue

    def total_records(self) -> int:
        count = 0
        for f in os.listdir(self.data_dir):
            if f.endswith(".jsonl"):
                with open(os.path.join(self.data_dir, f), encoding="utf-8") as chunk:
                    count += sum(1 for _ in chunk)
        return count
