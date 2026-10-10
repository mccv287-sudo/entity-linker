"""Tipos finos de GLiNER y su compatibilidad con las clases de Wikidata.

GLiNER detecta tipos finos ("sports team", "city"...), que se agrupan en el
esquema CoNLL (PER, ORG, LOC, MISC) para la salida y la evaluación del NER,
pero se conservan en ``EntitySpan.fine_label`` como señal de desambiguación:
un "sports team" encaja con un candidato que desciende (P31/P279*) de la clase
Q12973014 y no con una ciudad.
"""

# Etiquetas genéricas: solo como referencia para comparar con los tipos finos
GENERIC_LABELS = {
    "person": "PER",
    "organization": "ORG",
    "location": "LOC",
    "miscellaneous entity": "MISC",
}

# Tipo fino pedido a GLiNER -> etiqueta CoNLL (el valor por defecto del NER)
FINE_LABELS = {
    "person": "PER",
    "sports team": "ORG",
    "company": "ORG",
    "political party": "ORG",
    "news agency": "ORG",
    "international organization": "ORG",
    "government agency": "ORG",
    "university": "ORG",
    "organization": "ORG",
    "country": "LOC",
    "city": "LOC",
    "region": "LOC",
    "location": "LOC",
    "nationality": "MISC",
    "sports competition": "MISC",
    "event": "MISC",
    "award": "MISC",
    "miscellaneous entity": "MISC",
}

# Tipo fino -> clase raíz de Wikidata. Sin entrada: no se filtra (tipo genérico)
FINE_TYPE_CLASSES = {
    "person": "Q5",  # human
    "sports team": "Q12973014",
    "company": "Q4830453",  # business
    "political party": "Q7278",
    "news agency": "Q192283",
    "international organization": "Q484652",
    "government agency": "Q327333",
    "university": "Q3918",
    "organization": "Q43229",
    "country": "Q6256",
    "nationality": "Q6256",  # AIDA enlaza los gentilicios al país ("Chinese" -> China)
    "city": "Q515",
    "region": "Q82794",
    "location": "Q2221906",  # geographic location
    "sports competition": "Q13406554",
    "event": "Q1190554",  # occurrence
    "award": "Q618779",
}

ROOT_CLASSES = set(FINE_TYPE_CLASSES.values())  # las únicas que se piden a Wikidata

Classes = dict[str, tuple[set[str], set[str]]]  # QID -> (P31 directas, raíces)
