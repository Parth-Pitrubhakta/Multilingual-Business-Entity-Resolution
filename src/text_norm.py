"""Text normalisation for business names and addresses.

Everything here is country-agnostic: the dictionaries cover legal forms,
street-type abbreviations and state names for several countries, and anything
unknown simply passes through as a lower-cased, accent-stripped token. The
only learned resource is the non-Latin -> Latin name-token dictionary built by
`translit.py` from the training pairs.
"""
import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------
# Static dictionaries
# --------------------------------------------------------------------------

# Native-script Indian state names seen in Source 2/3 addresses.
NATIVE_STATES = {
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "उत्तर प्रदेश": "uttar pradesh",
    "ಕರ್ನಾಟಕ": "karnataka", "தமிழ்நாடு": "tamil nadu", "পশ্চিমবঙ্গ": "west bengal",
    "ગુજરાત": "gujarat", "తెలంగాణ": "telangana", "हरियाणा": "haryana",
    "राजस्थान": "rajasthan", "കേരളം": "kerala", "बिहार": "bihar",
    "मध्य प्रदेश": "madhya pradesh", "ఆంధ్రప్రదేశ్": "andhra pradesh",
    "ਪੰਜਾਬ": "punjab", "ଓଡ଼ିଶା": "odisha",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "ladakh": "la", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "andaman and nicobar islands": "an", "lakshadweep": "ld",
}
# French regions and their departments (departments map to their region's key),
# so that 'Bordeaux, Nouvelle-Aquitaine' and 'BORDEAUX, Gironde' agree on state.
FR_REGIONS = {
    "ara": ["Auvergne-Rhône-Alpes", "Ain", "Allier", "Ardèche", "Cantal", "Drôme", "Isère", "Loire",
            "Haute-Loire", "Puy-de-Dôme", "Rhône", "Savoie", "Haute-Savoie"],
    "bfc": ["Bourgogne-Franche-Comté", "Côte-d'Or", "Doubs", "Jura", "Nièvre", "Haute-Saône",
            "Saône-et-Loire", "Yonne", "Territoire de Belfort"],
    "bre": ["Bretagne", "Côtes-d'Armor", "Finistère", "Ille-et-Vilaine", "Morbihan"],
    "cvl": ["Centre-Val de Loire", "Cher", "Eure-et-Loir", "Indre", "Indre-et-Loire", "Loir-et-Cher", "Loiret"],
    "cor": ["Corse", "Corse-du-Sud", "Haute-Corse"],
    "ges": ["Grand Est", "Ardennes", "Aube", "Marne", "Haute-Marne", "Meurthe-et-Moselle", "Meuse", "Moselle",
            "Bas-Rhin", "Haut-Rhin", "Vosges"],
    "hdf": ["Hauts-de-France", "Aisne", "Nord", "Oise", "Pas-de-Calais", "Somme"],
    "idf": ["Île-de-France", "Paris", "Seine-et-Marne", "Yvelines", "Essonne", "Hauts-de-Seine",
            "Seine-Saint-Denis", "Val-de-Marne", "Val-d'Oise"],
    "nor": ["Normandie", "Calvados", "Eure", "Manche", "Orne", "Seine-Maritime"],
    "naq": ["Nouvelle-Aquitaine", "Charente", "Charente-Maritime", "Corrèze", "Creuse", "Dordogne", "Gironde",
            "Landes", "Lot-et-Garonne", "Pyrénées-Atlantiques", "Deux-Sèvres", "Vienne", "Haute-Vienne"],
    "occ": ["Occitanie", "Ariège", "Aude", "Aveyron", "Gard", "Haute-Garonne", "Gers", "Hérault", "Lot", "Lozère",
            "Hautes-Pyrénées", "Pyrénées-Orientales", "Tarn", "Tarn-et-Garonne"],
    "pdl": ["Pays de la Loire", "Loire-Atlantique", "Maine-et-Loire", "Mayenne", "Sarthe", "Vendée"],
    "pac": ["Provence-Alpes-Côte d'Azur", "Alpes-de-Haute-Provence", "Hautes-Alpes", "Alpes-Maritimes",
            "Bouches-du-Rhône", "Var", "Vaucluse"],
    "gp": ["Guadeloupe"], "mq": ["Martinique"], "gf": ["Guyane"], "re": ["La Réunion", "Réunion"], "yt": ["Mayotte"],
}
# Codes that appear as abbreviations in Indian records but differ from ISO.
IN_CODE_ALIASES = {"ts": "tg", "or": "od", "ori": "od", "chh": "cg", "ct": "cg", "uk": "uk", "ua": "uk"}

# Legal forms: every variant maps to one canonical token.
LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "pvte": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "limitada": "ltd",
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cos": "co", "companies": "co", "compagnie": "cie", "cie": "cie",
    "llc": "llc", "llp": "llp", "lp": "lp", "lllp": "lllp", "pllc": "pllc", "plc": "plc",
    "pc": "pc", "pa": "pa", "ltee": "ltd", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "scp": "scp", "sca": "sca", "sel": "sel", "selarl": "selarl",
    "ei": "ei", "eirl": "eirl", "sem": "sem", "gie": "gie", "scm": "scm",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}
# Name-body abbreviations (not legal forms).
NAME_ABBR = {
    "intl": "international", "int'l": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "services", "assoc": "associates", "assocs": "associates",
    "bros": "brothers", "mgmt": "management", "natl": "national", "ctr": "center",
    "centre": "center", "cntr": "center", "tech": "technologies", "techs": "technologies",
    "technology": "technologies", "grp": "group", "hldgs": "holdings", "hldg": "holding",
    "dev": "development", "devs": "developers", "engg": "engineering", "eng": "engineering",
    "univ": "university", "hosp": "hospital", "ent": "enterprises", "entp": "enterprises",
    "inds": "industries", "ind": "industries", "industry": "industries", "sys": "systems",
    "soln": "solutions", "solns": "solutions", "st": "saint", "ste": "societe",
    "soc": "societe", "sté": "societe", "ets": "etablissements", "etabl": "etablissements",
}
# Tokens that carry no identity in names (honorifics, articles, conjunctions).
NAME_STOP = {
    "the", "and", "of", "a", "an", "m", "s", "ms", "mr", "mrs", "dr", "shri", "sri", "shree",
    "smt", "de", "du", "des", "la", "le", "les", "et", "l", "d", "en",
    "india", "france", "usa", "us", "america", "of", "for", "by", "at", "in",
}
DBA_PAT = re.compile(
    r"\b(?:doing business as|d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|trading as|also known as|aka|formerly known as|fka)\b",
    re.I,
)

# Address abbreviations -> canonical (short) form. Both sides of a pair are
# normalised with the same table, so the choice of canonical form is arbitrary.
ADDR_ABBR = {
    # English street types
    "street": "st", "str": "st", "st": "st", "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr",
    "drv": "dr", "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave", "aven": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "crt": "ct", "place": "pl", "pl": "pl",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy", "highway": "hwy", "hwy": "hwy",
    "circle": "cir", "cir": "cir", "terrace": "ter", "ter": "ter", "terr": "ter",
    "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq", "way": "way", "wy": "way",
    "point": "pt", "pt": "pt", "ridge": "rdg", "rdg": "rdg", "run": "run", "pike": "pike",
    "crossing": "xing", "xing": "xing", "cove": "cv", "cv": "cv", "loop": "loop", "path": "path",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy", "turnpike": "tpke",
    "tpke": "tpke", "alley": "aly", "aly": "aly", "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "heights": "hts", "hts": "hts", "hill": "hl", "hl": "hl", "hills": "hls", "hls": "hls",
    "mount": "mt", "mt": "mt", "mountain": "mtn", "mtn": "mtn", "fort": "ft", "ft": "ft",
    "junction": "jct", "jct": "jct", "estate": "est", "estates": "ests", "plaza": "plz", "plz": "plz",
    "apartment": "apt", "apartments": "apts", "apt": "apt", "appt": "apt", "appartement": "apt",
    "suite": "ste", "ste": "ste", "floor": "fl", "fl": "fl", "flr": "fl", "unit": "unit",
    "building": "bldg", "bldg": "bldg", "room": "rm", "rm": "rm", "department": "dept",
    "north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e", "w": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "so": "s", "no": "no", "number": "no", "num": "no", "nr": "near", "near": "near",
    "opp": "opp", "opposite": "opp", "po": "po", "box": "box", "saint": "st", "sainte": "ste",
    "city": "city", "county": "cty", "cty": "cty", "twp": "twp", "township": "twp",
    # Indian address vocabulary
    "marg": "marg", "nagar": "nagar", "ngr": "nagar", "colony": "colony", "col": "colony",
    "district": "dist", "dist": "dist", "distt": "dist", "dt": "dist", "taluk": "tq", "taluka": "tq",
    "tq": "tq", "tal": "tq", "tehsil": "teh", "teh": "teh", "village": "vill", "vill": "vill",
    "vil": "vill", "post": "po", "sector": "sec", "sec": "sec", "sect": "sec", "phase": "ph", "ph": "ph",
    "plot": "plot", "plt": "plot", "house": "h", "h": "h", "hno": "h", "flat": "flat",
    "shop": "shop", "industrial": "indl", "indl": "indl", "indus": "indl", "indi": "indl",
    "area": "area", "extension": "extn", "extn": "extn", "ext": "extn", "layout": "layout",
    "cross": "cross", "main": "main", "block": "blk", "blk": "blk", "ward": "ward",
    "mandal": "mandal", "survey": "sy", "sy": "sy", "khasra": "khasra", "kh": "khasra",
    "chs": "chs", "society": "soc", "soc": "soc", "complex": "cmplx", "cmplx": "cmplx",
    "tower": "twr", "twr": "twr", "towers": "twr", "ground": "gnd", "gr": "gnd", "grd": "gnd",
    "gf": "gnd", "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11", "twelfth": "12",
    # French address vocabulary
    "rue": "rue", "r": "rue", "avenue": "ave", "allee": "all", "all": "all", "impasse": "imp",
    "imp": "imp", "chemin": "chem", "chem": "chem", "ch": "chem", "route": "rte", "rte": "rte",
    "quai": "qu", "qu": "qu", "cours": "crs", "crs": "crs", "faubourg": "fbg", "fbg": "fbg",
    "residence": "res", "res": "res", "lieu": "ld", "lieudit": "ld", "ld": "ld",
    "hameau": "ham", "ham": "ham", "passage": "pass", "pass": "pass", "sentier": "sen",
    "promenade": "prom", "prom": "prom", "rond": "rpt", "rondpoint": "rpt", "rpt": "rpt",
    "carrefour": "car", "voie": "voie", "cite": "cite", "zone": "zone", "za": "za", "zi": "zi",
    "zac": "zac", "bat": "bldg", "batiment": "bldg", "etage": "fl", "esc": "esc",
    "escalier": "esc", "numero": "no", "bis": "bis", "ter": "ter", "quater": "quater",
    "lotissement": "lot", "lot": "lot", "mail": "mail", "parvis": "parvis", "esplanade": "espl",
}
# Tokens that are pure noise in addresses (placeholder values).
ADDR_NOISE = {"null", "none", "nan", "na", "n/a", "nil", "unknown", "tbd", ""}

_ALNUM_SPLIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th|er|e|eme|ere)$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WEB = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co\.in|co|fr|biz|info|us|io|org\.in|net\.in)\b")
_LATIN_OK = re.compile(r"^[\x00-\x7fÀ-ɏ‘-‟°º]*$")
_VOWELS = re.compile(r"[aeiouy]")
_REPEAT = re.compile(r"(.)\1+")


def ascii_fold(s: str) -> str:
    """Strip accents / transliterate to ASCII and lower-case."""
    s = s.replace("Â", " ").replace("â\u0080\u0099", "'")  # mojibake 'Â'
    s = s.replace("N°", " numero ").replace("Nº", " numero ").replace("n°", " numero ").replace("nº", " numero ")
    return unidecode(s).lower()


def skeleton(tok: str) -> str:
    """Consonant skeleton: drop vowels (keep first char), collapse repeats.

    Makes transliteration variants (e.g. unidecoded 'limittedd' vs 'limited')
    and many typos collapse to the same key.
    """
    if not tok:
        return tok
    t = _REPEAT.sub(r"\1", tok)
    return t[0] + _VOWELS.sub("", t[1:])


# --------------------------------------------------------------------------
# Name normalisation
# --------------------------------------------------------------------------

class NameNormalizer:
    """Turns a raw business name into several comparable representations."""

    # Abbreviated legal forms that always co-occur, so EM alignment cannot
    # separate them ("प्रा. लि." = "Pvt. Ltd.").
    TRANSLIT_OVERRIDE = {"प्रा": "pvt", "लि": "ltd", "ప్రై": "pvt", "లి": "ltd", "ಪ್ರೈ": "pvt", "ಲಿ": "ltd",
                         "প্রা": "pvt", "লি": "ltd", "પ્રા": "pvt", "લિ": "ltd", "பி": "pvt", "லி": "ltd"}

    def __init__(self, translit: dict | None = None):
        # translit: raw non-Latin token -> Latin token (learned from training pairs)
        self.translit = dict(translit or {})
        self.translit.update(self.TRANSLIT_OVERRIDE)

    def _translit_tokens(self, raw: str) -> str:
        """Replace non-Latin words using the learned dictionary (unidecode fallback)."""
        if _LATIN_OK.match(raw):
            return raw
        out = []
        for w in raw.split():
            if _LATIN_OK.match(w):
                out.append(w)
                continue
            key = w.strip(".,()[]-")
            rep = self.translit.get(key)
            if rep is None:
                rep = _REPEAT.sub(r"\1", unidecode(key))
            out.append(rep)
        return " ".join(out)

    @staticmethod
    def _merge_initials(tokens):
        """Merge runs of >=2 single letters: 's a r l' -> 'sarl', 'p c' -> 'pc'."""
        out, run = [], []
        for t in tokens:
            if len(t) == 1 and t.isalpha():
                run.append(t)
                continue
            if run:
                out.extend(["".join(run)] if len(run) >= 2 else run)
                run = []
            out.append(t)
        if run:
            out.extend(["".join(run)] if len(run) >= 2 else run)
        return out

    def _tokens(self, s: str):
        """Lower-case ASCII tokens with legal/abbreviation canonicalisation."""
        s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
        s = s.replace("'", "").replace("’", "")
        toks = []
        for seg in s.split(","):
            seg = seg.replace(".", " ")
            seg_toks = [t for t in _NON_ALNUM.split(seg) if t]
            toks.extend(self._merge_initials(seg_toks))
        out = []
        for t in toks:
            t = NAME_ABBR.get(t, t)
            t = LEGAL.get(t, t)
            out.append(t)
        return out

    def normalize(self, name: str | None) -> dict:
        if not name:
            name = ""
        raw = unicodedata.normalize("NFC", name)
        has_nonlatin = not _LATIN_OK.match(raw)
        raw = self._translit_tokens(raw)
        s = ascii_fold(raw)

        # Websites: 'www.acme.com' -> stem 'acme'
        web_stems = _WEB.findall(s)
        is_web = bool(web_stems)
        s_noweb = _WEB.sub(" ", s)

        # DBA / trade names split the string into alternatives.
        parts = [p for p in DBA_PAT.split(s_noweb.replace("|", " dba ")) if p.strip()]
        main = parts[0] if parts else s_noweb
        alt = " ".join(parts[1:]) if len(parts) > 1 else ""

        toks = self._tokens(s_noweb.replace("|", " "))
        toks = [t for t in toks if t not in ("dba", "aka", "doing", "business", "as")] if (alt or "dba" in toks) else toks
        legal = sorted({t for t in toks if t in LEGAL.values()})
        core = [t for t in toks if t not in LEGAL.values() and t not in NAME_STOP]
        if not core and web_stems:  # e.g. 'Sri vallabhsons.com' -> core 'vallabhsons'
            core = list(web_stems)
        if not core:  # e.g. name is only a legal form / stop words
            core = [t for t in toks if t not in LEGAL.values()] or toks
        main_core = [t for t in self._tokens(main) if t not in LEGAL.values() and t not in NAME_STOP]
        alt_core = [t for t in self._tokens(alt) if t not in LEGAL.values() and t not in NAME_STOP] if alt else []

        web = " ".join(web_stems)
        return {
            "n_full": " ".join(toks) if toks else web,
            "n_core": " ".join(core),
            "n_main": " ".join(main_core) if main_core else " ".join(core),
            "n_alt": " ".join(alt_core),
            "n_legal": " ".join(legal),
            "n_web": web,
            "n_skel": " ".join(skeleton(t) for t in core),
            "n_acr": "".join(t[0] for t in core if t and t[0].isalpha()),
            "n_nonlatin": has_nonlatin,
            "n_isweb": is_web,
        }


# --------------------------------------------------------------------------
# Address normalisation
# --------------------------------------------------------------------------

_STATE_LOOKUP = {}
for _full, _code in US_STATES.items():
    _STATE_LOOKUP[_full] = "us_" + _code
for _full, _code in IN_STATES.items():
    _STATE_LOOKUP[_full] = "in_" + _code
_FR_LOOKUP = {}
for _code, _names in FR_REGIONS.items():
    for _n in _names:
        _FR_LOOKUP[" ".join(t for t in _NON_ALNUM.split(unidecode(_n).lower().replace("'", " ")) if t)] = "fr_" + _code
_US_CODES = {v: "us_" + v for v in US_STATES.values()}
_IN_CODES = {v: "in_" + v for v in IN_STATES.values()}
_IN_CODES.update({k: "in_" + v for k, v in IN_CODE_ALIASES.items()})


def _state_of_component(comp: str, country: str):
    """Canonical state key if the whole comma-component is a state name/code."""
    if comp in _STATE_LOOKUP:
        key = _STATE_LOOKUP[comp]
        # 'washington' / 'delhi' are also city names; accept either way.
        return key
    c = country.lower() if country else ""
    if c == "france" and comp in _FR_LOOKUP:
        return _FR_LOOKUP[comp]
    if len(comp) in (2, 3):
        if c == "us" and comp in _US_CODES:
            return _US_CODES[comp]
        if c == "india" and comp in _IN_CODES:
            return _IN_CODES[comp]
    return None


def _num_norm(t: str) -> str:
    """Strip leading zeros of a digit string ('00380' -> '380')."""
    s = t.lstrip("0")
    return s if s else "0"


def normalize_address(addr: str | None, country: str | None) -> dict:
    """Return tokens, numbers, state and postal code for an address string."""
    if not addr:
        addr = ""
    s = unicodedata.normalize("NFC", addr)
    for k, v in NATIVE_STATES.items():
        if k in s:
            s = s.replace(k, v)
    s = ascii_fold(s)
    s = s.replace("&", " and ").replace("'", " ").replace("#", " ")
    comps = [c.strip() for c in s.split(",")]
    states = []
    tokens = []
    nums = []
    comp_keys = []
    for comp in comps:
        comp_clean = " ".join(t for t in _NON_ALNUM.split(comp) if t)
        if comp_clean in ADDR_NOISE:
            continue
        st = _state_of_component(comp_clean, country or "")
        if st is not None:
            states.append(st)
            continue
        ctoks = []
        for raw_t in comp_clean.split():
            m = _ORDINAL.match(raw_t)
            if m:
                ctoks.append(_num_norm(m.group(1)))
                continue
            for t in _ALNUM_SPLIT.split(raw_t):
                if t.isdigit():
                    ctoks.append(_num_norm(t))
                elif t in ADDR_NOISE:
                    continue
                else:
                    ctoks.append(ADDR_ABBR.get(t, t))
        if ctoks:
            comp_keys.append(" ".join(ctoks))
            tokens.extend(ctoks)
    for t in tokens:
        if t.isdigit():
            nums.append(t)
    # Postal code: a 5-6 digit number that does not start its component
    # (a leading number is usually the house number).
    postal = ""
    for ck in comp_keys:
        for t in ck.split()[1:]:
            if t.isdigit() and len(t) in (5, 6):
                postal = t
    # Heuristic "last word-only component" is often the city.
    city = ""
    for ck in reversed(comp_keys):
        if not any(ch.isdigit() for ch in ck):
            city = ck
            break
    return {
        "a_tok": " ".join(tokens),
        "a_nums": " ".join(nums),
        "a_state": " ".join(sorted(set(states))),
        "a_postal": postal,
        "a_city": city,
        "a_comps": "|".join(comp_keys),
    }
