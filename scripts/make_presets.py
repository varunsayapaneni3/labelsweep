"""Regenerate the bundled label presets (src/labelsweep/presets/*.yaml).

The wordings below were written on 2026-11-14 BEFORE any model was run on these datasets
(pre-registration); do not tune them on results. Per type: natural default, two hand-written
alternatives, the raw dataset tag (if it differs), a casing variant, and the plural.
"""

from pathlib import Path

import yaml

from labelsweep.sweep import pluralize

NATURAL = {
    "academicjournal": ("academic journal", ["scientific journal", "journal"]),
    "album": ("album", ["music album", "record"]),
    "algorithm": ("algorithm", ["machine learning algorithm", "method"]),
    "astronomicalobject": ("astronomical object", ["celestial body", "planet or star"]),
    "award": ("award", ["prize", "honor"]),
    "band": ("band", ["music group", "musical band"]),
    "book": ("book", ["literary work", "novel"]),
    "chemicalcompound": ("chemical compound", ["compound", "chemical substance"]),
    "chemicalelement": ("chemical element", ["element", "periodic table element"]),
    "conference": ("conference", ["academic conference", "scientific venue"]),
    "country": ("country", ["nation", "sovereign state"]),
    "discipline": ("discipline", ["academic discipline", "scientific field"]),
    "election": ("election", ["political election", "vote"]),
    "enzyme": ("enzyme", ["biological catalyst", "protein enzyme"]),
    "event": ("event", ["historical event", "occurrence"]),
    "field": ("field", ["research field", "area of study"]),
    "literarygenre": ("literary genre", ["genre", "literary form"]),
    "location": ("location", ["place", "geographic location"]),
    "magazine": ("magazine", ["periodical", "publication"]),
    "metrics": ("metric", ["evaluation metric", "performance measure"]),
    "misc": ("miscellaneous", ["other entity", "named entity"]),
    "musicalartist": ("musical artist", ["musician", "singer"]),
    "musicalinstrument": ("musical instrument", ["instrument"]),
    "musicgenre": ("music genre", ["musical style", "genre of music"]),
    "organisation": ("organization", ["company", "institution"]),
    "person": ("person", ["person name", "human"]),
    "poem": ("poem", ["poetry", "verse work"]),
    "politicalparty": ("political party", ["party", "political organization"]),
    "politician": ("politician", ["political figure", "elected official"]),
    "product": ("product", ["software product", "commercial product"]),
    "programlang": ("programming language", ["coding language", "language"]),
    "protein": ("protein", ["gene product", "biomolecule"]),
    "researcher": ("researcher", ["scientist", "academic"]),
    "scientist": ("scientist", ["researcher", "scholar"]),
    "song": ("song", ["track", "single"]),
    "task": ("task", ["machine learning task", "research problem"]),
    "theory": ("theory", ["scientific theory", "hypothesis"]),
    "university": ("university", ["college", "academic institution"]),
    "writer": ("writer", ["author", "novelist"]),
    # MIT restaurant
    "Amenity": ("amenity", ["restaurant feature", "service"]),
    "Cuisine": ("cuisine", ["type of food", "food style"]),
    "Dish": ("dish", ["food item", "meal"]),
    "Hours": ("hours", ["opening hours", "time"]),
    "Location": ("location", ["place", "area"]),
    "Price": ("price", ["price range", "cost"]),
    "Rating": ("rating", ["review rating", "quality"]),
    "Restaurant_Name": ("restaurant name", ["restaurant", "business name"]),
}

DATASETS = {
    "crossner_ai": [
        "task",
        "field",
        "product",
        "metrics",
        "misc",
        "algorithm",
        "researcher",
        "organisation",
        "conference",
        "person",
        "programlang",
        "country",
        "location",
        "university",
    ],
    "crossner_literature": [
        "writer",
        "book",
        "misc",
        "literarygenre",
        "person",
        "award",
        "poem",
        "organisation",
        "country",
        "location",
        "magazine",
        "event",
    ],
    "crossner_music": [
        "musicalartist",
        "musicgenre",
        "band",
        "location",
        "album",
        "award",
        "song",
        "organisation",
        "misc",
        "country",
        "person",
        "event",
        "musicalinstrument",
    ],
    "crossner_politics": [
        "politicalparty",
        "location",
        "organisation",
        "politician",
        "election",
        "country",
        "person",
        "misc",
        "event",
    ],
    "crossner_science": [
        "misc",
        "scientist",
        "astronomicalobject",
        "organisation",
        "person",
        "academicjournal",
        "location",
        "chemicalcompound",
        "protein",
        "award",
        "university",
        "chemicalelement",
        "enzyme",
        "discipline",
        "event",
        "country",
        "theory",
    ],
    "mit_restaurant": [
        "Amenity",
        "Cuisine",
        "Dish",
        "Hours",
        "Location",
        "Price",
        "Rating",
        "Restaurant_Name",
    ],
}

HEADER = (
    "# labelsweep preset: gold type -> candidate wordings (first = default).\n"
    "# Pre-registered 2026-11-14 before any model was run on this dataset (scripts/make_presets.py).\n"
    "# Per type: natural default, 2 hand-written alternatives, raw dataset tag (if different), casing, plural.\n"
)


def candidates(tag: str) -> list[str]:
    default, alts = NATURAL[tag]
    out = [default, *alts]
    raw = tag.lower().replace("_", " ")
    if raw not in out:
        out.append(raw)
    out.append(default.title() if default.title() != default else default.upper())
    if (pl := pluralize(default)) is not None:
        out.append(pl)
    return list(dict.fromkeys(out))


def main() -> None:
    outdir = Path(__file__).resolve().parents[1] / "src" / "labelsweep" / "presets"
    outdir.mkdir(exist_ok=True)
    for name, tags in DATASETS.items():
        owner: dict[str, str] = {}
        data = {}
        for t in tags:
            keep = []
            for w in candidates(t):  # a wording may belong to only one type per file
                if owner.setdefault(w.lower(), t) == t:
                    keep.append(w)
            data[t] = keep
        (outdir / f"{name}.yaml").write_text(
            HEADER + yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        print(name, sum(len(v) for v in data.values()), "wordings,", len(data), "types")


if __name__ == "__main__":
    main()
