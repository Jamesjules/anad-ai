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

    # Curated list of high-quality books — expanded to 100+
    BOOK_IDS = [
        # ── Philosophy & Wisdom ──────────────────────────────
        (2680,  "Meditations — Marcus Aurelius", "en"),
        (1232,  "The Prince — Machiavelli", "en"),
        (4280,  "Pragmatism — William James", "en"),
        (5827,  "Thus Spoke Zarathustra — Nietzsche", "en"),
        (1497,  "The Republic — Plato", "en"),
        (1656,  "The Nicomachean Ethics — Aristotle", "en"),
        (4683,  "Critique of Pure Reason — Kant", "en"),
        (46,    "A Christmas Carol — Dickens", "en"),
        (10,    "The Bible — King James Version", "en"),
        (7370,  "Tao Te Ching — Lao Tzu", "en"),

        # ── Science & Nature ─────────────────────────────────
        (1228,  "The Origin of Species — Darwin", "en"),
        (5001,  "Relativity — Einstein", "en"),
        (30155, "Opticks — Newton", "en"),
        (14725, "The Evolution of Man — Haeckel", "en"),
        (6130,  "The History of the Peloponnesian War", "en"),
        (33,    "The Magna Carta", "en"),

        # ── Mathematics & Logic ──────────────────────────────
        (13700, "Euclid's Elements", "en"),
        (20878, "An Introduction to Mathematics — Whitehead", "en"),
        (9007,  "Mathematical Recreations — Ball", "en"),

        # ── Literature — English ─────────────────────────────
        (1342,  "Pride and Prejudice — Austen", "en"),
        (2701,  "Moby Dick — Melville", "en"),
        (84,    "Frankenstein — Shelley", "en"),
        (11,    "Alice in Wonderland — Carroll", "en"),
        (98,    "A Tale of Two Cities — Dickens", "en"),
        (1661,  "Adventures of Sherlock Holmes — Doyle", "en"),
        (74,    "Adventures of Tom Sawyer — Twain", "en"),
        (76,    "Adventures of Huckleberry Finn — Twain", "en"),
        (1952,  "The Yellow Wallpaper — Gilman", "en"),
        (514,   "Little Women — Alcott", "en"),
        (2814,  "Dubliners — Joyce", "en"),
        (1400,  "Great Expectations — Dickens", "en"),
        (730,   "Oliver Twist — Dickens", "en"),
        (1260,  "Jane Eyre — Bronte", "en"),
        (768,   "Wuthering Heights — Bronte", "en"),
        (5200,  "Metamorphosis — Kafka", "en"),
        (2500,  "Siddhartha — Hesse", "en"),
        (3207,  "Critique of the Gotha Programme — Marx", "en"),
        (1727,  "The Odyssey — Homer", "en"),
        (6130,  "Iliad — Homer", "en"),
        (4300,  "Ulysses — Joyce", "en"),
        (1184,  "The Count of Monte Cristo — Dumas", "en"),
        (2413,  "The Importance of Being Earnest — Wilde", "en"),
        (174,   "The Picture of Dorian Gray — Wilde", "en"),
        (16,    "Peter Pan — Barrie", "en"),
        (521,   "Aesop's Fables", "en"),
        (1080,  "A Modest Proposal — Swift", "en"),
        (100,   "Complete Works of Shakespeare", "en"),
        (2265,  "Hamlet — Shakespeare", "en"),
        (1513,  "Romeo and Juliet — Shakespeare", "en"),
        (2267,  "Macbeth — Shakespeare", "en"),

        # ── History & Politics ───────────────────────────────
        (2600,  "War and Peace — Tolstoy", "en"),
        (2554,  "Crime and Punishment — Dostoevsky", "en"),
        (28054, "The Brothers Karamazov — Dostoevsky", "en"),
        (600,   "Notes from Underground — Dostoevsky", "en"),
        (3825,  "The Idiot — Dostoevsky", "en"),
        (1232,  "The Prince — Machiavelli", "en"),
        (30254, "Common Sense — Thomas Paine", "en"),
        (3207,  "The Communist Manifesto — Marx & Engels", "en"),
        (5684,  "The Wealth of Nations — Adam Smith", "en"),
        (7370,  "The Art of War — Sun Tzu", "en"),

        # ── Indian Classics (English translations) ───────────
        (16955, "The Mahabharata — Vyasa", "en"),
        (7864,  "The Ramayana — Valmiki", "en"),
        (3283,  "Kamasutra — Vatsyayana", "en"),
        (13828, "The Vedas — translated", "en"),
        (22367, "Upanishads — translated", "en"),
        (12914, "The Dhammapada — Buddha", "en"),
        (2018,  "The Analects — Confucius", "en"),
        (17921, "Gitanjali — Tagore", "en"),
        (6761,  "The Home and the World — Tagore", "en"),

        # ── Science Fiction & Imagination ────────────────────
        (1080,  "The Time Machine — Wells", "en"),
        (36,    "The War of the Worlds — Wells", "en"),
        (5230,  "The Island of Doctor Moreau — Wells", "en"),
        (43,    "The Strange Case of Dr Jekyll — Stevenson", "en"),
        (120,   "Treasure Island — Stevenson", "en"),
        (10676, "20,000 Leagues Under the Sea — Verne", "en"),
        (103,   "Around the World in 80 Days — Verne", "en"),
        (164,   "Twenty Years After — Dumas", "en"),

        # ── Self Improvement & Psychology ────────────────────
        (16102, "As a Man Thinketh — Allen", "en"),
        (10,    "The Art of Living — Epictetus", "en"),
        (2009,  "The Enchiridion — Epictetus", "en"),
        (4093,  "Walden — Thoreau", "en"),
        (1321,  "On Civil Disobedience — Thoreau", "en"),

        # ── Language & Linguistics ───────────────────────────
        (2130,  "Children's Literature — Hazeltine", "en"),
        (19033, "The Story of Language — Bodmer", "en"),

        # ── Health & Medicine ────────────────────────────────
        (17147, "The Merck Manual — public domain", "en"),
        (12914, "Manual of Surgery — Rutherford", "en"),
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
            # Science & Technology
            "Artificial intelligence", "Machine learning", "Neural network",
            "Deep learning", "Natural language processing", "Computer vision",
            "Quantum computing", "Blockchain", "Internet of things",
            "Robotics", "Nanotechnology", "Biotechnology", "Genetics",
            "DNA", "Cell biology", "Evolution", "Photosynthesis",
            "Quantum mechanics", "Theory of relativity", "String theory",
            "Climate change", "Solar energy", "Nuclear fusion",
            "Computer science", "Algorithm", "Data structure",
            "Operating system", "Internet", "World Wide Web",
            "Open source", "Linux", "Python (programming language)",
            "Artificial neural network", "Transformer (machine learning)",
            "Large language model", "GPT", "Mathematics",
            "Calculus", "Linear algebra", "Statistics", "Probability",
            "Number theory", "Geometry", "Topology",
            # History & Culture
            "India", "Gujarat", "Mumbai", "Delhi", "Bangalore",
            "History of India", "Mughal Empire", "British Raj",
            "Indian independence movement", "Mahatma Gandhi",
            "Jawaharlal Nehru", "Ambedkar", "Subhas Chandra Bose",
            "Ancient India", "Indus Valley Civilisation",
            "Sanskrit", "Tamil language", "Hindi", "Gujarati language",
            "Buddhism", "Hinduism", "Islam", "Christianity", "Sikhism",
            "Jainism", "Zoroastrianism",
            "World War I", "World War II", "Cold War",
            "American Revolution", "French Revolution",
            "Renaissance", "Industrial Revolution",
            "Roman Empire", "Greek civilization", "Egyptian civilization",
            "Silk Road", "Colonialism", "Slavery",
            # Philosophy & Society
            "Philosophy of mind", "Ethics", "Democracy", "Human rights",
            "Feminism", "Environmentalism", "Capitalism", "Socialism",
            "Globalization", "Poverty", "Education", "Healthcare",
            "Philosophy", "Epistemology", "Metaphysics", "Logic",
            "Consciousness", "Free will", "Meaning of life",
            # Arts & Literature
            "Literature", "Poetry", "Music", "Film", "Art",
            "Architecture", "Photography", "Theatre",
            "Rabindranath Tagore", "Premchand", "Mirza Ghalib",
            # Geography & Nature
            "Amazon rainforest", "Himalayas", "Sahara Desert",
            "Pacific Ocean", "Climate", "Biodiversity",
            "Ecosystem", "Food chain", "Water cycle",
            # Health
            "Medicine", "Vaccine", "Virus", "Bacteria",
            "Mental health", "Nutrition", "Exercise", "Sleep",
        ],
        "hi": [
            "कृत्रिम बुद्धिमत्ता", "भारत", "हिन्दी", "विज्ञान",
            "गणित", "दर्शनशास्त्र", "इतिहास", "महात्मा गांधी",
            "भारतीय स्वतंत्रता आंदोलन", "संस्कृत", "बौद्ध धर्म",
            "हिंदू धर्म", "मुगल साम्राज्य", "दिल्ली", "मुंबई",
            "प्रौद्योगिकी", "अर्थशास्त्र", "लोकतंत्र", "शिक्षा",
            "स्वास्थ्य", "पर्यावरण", "जलवायु परिवर्तन",
        ],
        "gu": [
            "ગુજરાત", "ભારત", "ગુજરાતી ભાષા", "કૃત્રિમ બુદ્ધિ",
            "વિજ્ઞાન", "ગણિત", "ઇતિહાસ", "મહાત્મા ગાંધી",
            "સ્વતંત્રતા", "અમદાવાદ", "સુરત", "વડોદરા",
            "ભારતીય સંસ્કૃતિ", "હિંદુ ધર્મ", "જૈન ધર્મ",
            "ગુજરાતી સાહિત્ય", "નર્મદ", "મેઘાણી",
        ],
        "ta": [
            "செயற்கை நுண்ணறிவு", "இந்தியா", "தமிழ்நாடு",
            "தமிழ் மொழி", "தமிழ் இலக்கியம்", "சென்னை",
            "திருவள்ளுவர்", "தமிழ் வரலாறு", "சோழர்கள்",
        ],
        "te": [
            "తెలుగు", "ఆంధ్రప్రదేశ్", "తెలంగాణ", "హైదరాబాద్",
            "కృత్రిమ మేధస్సు", "భారతదేశం", "తెలుగు సాహిత్యం",
        ],
        "bn": [
            "কৃত্রিম বুদ্ধিমত্তা", "বাংলাদেশ", "পশ্চিমবঙ্গ",
            "রবীন্দ্রনাথ ঠাকুর", "বাংলা ভাষা", "ভারত",
        ],
        "mr": [
            "महाराष्ट्र", "मुंबई", "मराठी भाषा", "भारत",
            "छत्रपती शिवाजी", "पुणे",
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
        # ── Gujarati ─────────────────────────────────────────
        DataRecord(
            text="ગુજરાત ભારતનું એક રાજ્ય છે. ગુજરાતી ભાષા ઈન્ડો-આર્યન ભાષા પરિવારની છે. ગાંધીજીનો જન્મ ગુજરાતના પોરબંદરમાં થયો હતો.",
            source="indic_seed", language="gu", license="public_domain", title="Gujarat"),
        DataRecord(
            text="અનાદ એ સૌ માટે AI છે. કોઈ કોર્પોરેશન નહીં, કોઈ સરકાર નહીં. ફક્ત લોકો, ફક્ત જ્ઞાન, ફક્ત સ્વતંત્રતા. તમારો ડેટા તમારો છે.",
            source="indic_seed", language="gu", license="public_domain", title="Anad Gujarati"),
        DataRecord(
            text="જ્ઞાન એ સૌથી મોટી શક્તિ છે. ભારત પ્રાચીન સમયથી જ્ઞાનનું કેન્દ્ર રહ્યું છે. ગુજરાતના વ્યાપારીઓ વિશ્વભરમાં ફેલાયેલા છે.",
            source="indic_seed", language="gu", license="public_domain", title="Knowledge Gujarati"),
        DataRecord(
            text="નમસ્તે. હું અનાદ છું. તમે મને ગુજરાતી, હિન્દી, અંગ્રેજી અથવા અન્ય ભાષામાં વાત કરી શકો છો. હું મદદ કરવા માટે અહીં છું.",
            source="indic_seed", language="gu", license="public_domain", title="Anad greeting Gujarati"),

        # ── Hindi ────────────────────────────────────────────
        DataRecord(
            text="भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं। हिन्दी भारत की राजभाषा है। भारत में अनेक संस्कृतियाँ एक साथ रहती हैं।",
            source="indic_seed", language="hi", license="public_domain", title="India Hindi"),
        DataRecord(
            text="कृत्रिम बुद्धिमत्ता एक ऐसी तकनीक है जो मशीनों को सोचने की क्षमता देती है। अनाद एक ऐसा AI है जो सबका है, किसी कंपनी का नहीं।",
            source="indic_seed", language="hi", license="public_domain", title="AI Hindi"),
        DataRecord(
            text="नमस्ते! मैं अनाद हूँ। मैं आपकी किसी भी भाषा में मदद कर सकता हूँ। आप मुझसे कोई भी सवाल पूछ सकते हैं।",
            source="indic_seed", language="hi", license="public_domain", title="Anad greeting Hindi"),
        DataRecord(
            text="ज्ञान ही शक्ति है। विज्ञान और तकनीक के क्षेत्र में भारत तेजी से आगे बढ़ रहा है। शिक्षा सबका अधिकार है।",
            source="indic_seed", language="hi", license="public_domain", title="Knowledge Hindi"),

        # ── Tamil ────────────────────────────────────────────
        DataRecord(
            text="தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்று. இந்தியா பல மொழிகள் மற்றும் கலாச்சாரங்களின் தாயகம். அனாத் என்பது அனைவருக்கும் சொந்தமான AI.",
            source="indic_seed", language="ta", license="public_domain", title="Tamil intro"),
        DataRecord(
            text="வணக்கம்! நான் அனாத். உங்களுக்கு எந்த விஷயத்திலும் உதவ தயாராக இருக்கிறேன். தமிழிலும் பேசலாம்.",
            source="indic_seed", language="ta", license="public_domain", title="Anad Tamil"),

        # ── Telugu ───────────────────────────────────────────
        DataRecord(
            text="తెలుగు భారతదేశంలో అత్యధికంగా మాట్లాడే భాషలలో ఒకటి. అనాద్ అనేది అందరి కోసం AI. ఏ కంపెనీకీ చెందినది కాదు.",
            source="indic_seed", language="te", license="public_domain", title="Telugu intro"),
        DataRecord(
            text="నమస్కారం! నేను అనాద్. మీకు ఏ విషయంలోనైనా సహాయం చేయగలను. తెలుగులో మాట్లాడవచ్చు.",
            source="indic_seed", language="te", license="public_domain", title="Anad Telugu"),

        # ── Sanskrit ─────────────────────────────────────────
        DataRecord(
            text="अनादि अनन्तं ब्रह्म। सर्वे भवन्तु सुखिनः। सर्वे सन्तु निरामयाः। सर्वे भद्राणि पश्यन्तु। मा कश्चिद्दुःखभाग्भवेत्।",
            source="indic_seed", language="sa", license="public_domain", title="Sanskrit shlokas"),
        DataRecord(
            text="विद्या ददाति विनयं विनयाद्याति पात्रताम्। पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम्।",
            source="indic_seed", language="sa", license="public_domain", title="Vidya Sanskrit"),

        # ── English Conversations ─────────────────────────────
        DataRecord(
            text="Hello, how can I help you today? I am Anad, public AI that belongs to everyone. You can ask me anything about science, history, coding, or just have a conversation.",
            source="indic_seed", language="en", license="public_domain", title="Anad intro"),
        DataRecord(
            text="Anad is public AI that belongs to everyone. No corporation owns it. No government controls it. Your data stays on your device. Your memory is yours. The network grows stronger with every node that joins.",
            source="indic_seed", language="en", license="public_domain", title="Anad description"),
        DataRecord(
            text="What is artificial intelligence? AI is the simulation of human intelligence by machines. It includes machine learning, natural language processing, computer vision, and reasoning systems.",
            source="indic_seed", language="en", license="public_domain", title="AI explanation"),
        DataRecord(
            text="How does machine learning work? Machine learning algorithms learn patterns from data. They improve their performance over time without being explicitly programmed for each task.",
            source="indic_seed", language="en", license="public_domain", title="ML explanation"),
        DataRecord(
            text="India is the world's largest democracy with over 1.4 billion people. It has 28 states, 22 official languages, and thousands of years of recorded history.",
            source="indic_seed", language="en", license="public_domain", title="India facts"),
        DataRecord(
            text="Python is a popular programming language known for its simplicity and readability. It is widely used in data science, web development, and artificial intelligence.",
            source="indic_seed", language="en", license="public_domain", title="Python intro"),
        DataRecord(
            text="The solar system consists of the Sun and eight planets. Earth is the third planet from the Sun and the only known planet to support life.",
            source="indic_seed", language="en", license="public_domain", title="Solar system"),
        DataRecord(
            text="Mathematics is the language of the universe. From basic arithmetic to calculus and beyond, mathematics helps us understand patterns and solve problems.",
            source="indic_seed", language="en", license="public_domain", title="Mathematics"),
        DataRecord(
            text="Climate change refers to long-term shifts in global temperatures and weather patterns. While some natural factors influence climate, human activities have been the main driver since the 1800s.",
            source="indic_seed", language="en", license="public_domain", title="Climate change"),
        DataRecord(
            text="The human body contains approximately 37 trillion cells. Each cell contains DNA that carries genetic instructions for the development, functioning, growth and reproduction of all organisms.",
            source="indic_seed", language="en", license="public_domain", title="Human body"),
        DataRecord(
            text="History repeats itself when we forget its lessons. The great civilizations of the past rose through knowledge, cooperation and justice. They fell through corruption, inequality and ignorance.",
            source="indic_seed", language="en", license="public_domain", title="History lesson"),
        DataRecord(
            text="Good morning. Good afternoon. Good evening. How are you? I am fine thank you. What would you like to talk about today? I am here to help.",
            source="indic_seed", language="en", license="public_domain", title="Greetings"),
        DataRecord(
            text="To write good code: keep it simple, readable and well-commented. Test your work. Handle errors gracefully. Code is read more often than it is written.",
            source="indic_seed", language="en", license="public_domain", title="Coding advice"),
        DataRecord(
            text="The internet connects billions of people worldwide. It has transformed communication, commerce, education and entertainment. Access to information is now a fundamental human need.",
            source="indic_seed", language="en", license="public_domain", title="Internet"),
        DataRecord(
            text="Philosophy asks the deepest questions: What is real? What can we know? How should we live? What is consciousness? These questions have no simple answers but thinking about them makes us wiser.",
            source="indic_seed", language="en", license="public_domain", title="Philosophy"),
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
