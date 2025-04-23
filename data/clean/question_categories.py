import random
import pprint

premise_categorization = {
    "categorization_name": "premise_categorization",
    "categories": [
        {
            "name": "without premise",
            "description": "a question that does not contain any premise or any information about the user.",
            "probability": 0.7
        },
        {
            "name": "with premise",
            "description": ("a question starting with a very short premise, where the users reveal "
                            "their needs or some information about themselves."),
            "probability": 0.3
        }
    ]
}

multilingual_categorization = {
    "categorization_name": "multilingual_categorization",
    "categories": [
        {
            "name": "monolingual",
            "description": "A question written entirely in one language, such as English, Spanish, or Chinese.",
            "probability": 0.6
        },
        {
            "name": "code-switched",
            "description": "A question that includes a mix of languages, where users switch between two or more languages in the same sentence or utterance.",
            "probability": 0.2
        },
        {
            "name": "cross-lingual-multisentence",
            "description": "A question composed of multiple sentences, each written in a different language — e.g., an English sentence followed by one in Spanish or Chinese.",
            "probability": 0.15
        },
        {
            "name": "language-ambiguous",
            "description": "A question where the language is unclear or contains multilingual borrowings, making it hard to classify as a single language.",
            "probability": 0.05
        }
    ]
}

question_length = {
    "categorization_name": "question_length",
    "categories": [
        {
            "name": "short",
            "description": "A brief question that is concise and to the point, often consisting of only a few words.",
            "probability": 0.3
        },
        {
            "name": "moderate",
            "description": "A question with a moderate amount of detail, offering some context but still relatively succinct.",
            "probability": 0.4
        },
        {
            "name": "long",
            "description": "An extended question that provides substantial context or includes multiple parts or clarifications.",
            "probability": 0.3
        }
    ]
}

answer_length = {
    "categorization_name": "answer_length",
    "categories": [
        {
            "name": "concise",
            "description": "Questions that imply brief, direct responses—often fact-based, binary (yes/no), or entity-level answers. Common in FAQs, product specs, or headlines.",
            "probability": 0.3
        },
        {
            "name": "moderate",
            "description": "Questions that invite short explanatory responses or summaries—typical of blog intros, how-to queries, or academic abstracts.",
            "probability": 0.4
        },
        {
            "name": "detailed",
            "description": "Questions suggesting elaborate, multi-sentence responses with reasoning, context, or analysis—frequent in academic discussions, long-form articles, or reviews.",
            "probability": 0.3
        }
    ]
}

question_formulation = {
    "categorization_name": "question_formulation",
    "categories": [
        {
            "name": "natural_simple",
            "description": "A concise, naturally phrased question seeking a straightforward fact or yes/no answer. Common in news headlines or product specs.",
            "probability": 0.25
        },
        {
            "name": "natural_synthesis",
            "description": "A naturally phrased question requiring integration of multiple data points—typical of blog posts or academic Q&A.",
            "probability": 0.20
        },
        {
            "name": "natural_inference",
            "description": "A natural-language question that demands reasoning or interpretation beyond explicit statements—often seen in editorials or opinion pieces.",
            "probability": 0.15
        },
        {
            "name": "search_simple",
            "description": "A short, keyword-only search-style query for fact lookup, as seen in FAQ pages or quick-reference tools.",
            "probability": 0.15
        },
        {
            "name": "search_synthesis",
            "description": "A longer keyword-based query implying comparison or synthesis (e.g., multiple keywords suggesting an analytical request).",
            "probability": 0.05
        },
        {
            "name": "format_constrained",
            "description": "Any question (natural or search) that explicitly demands output in a specific format (list, table, summary, code snippet).",
            "probability": 0.08
        },
        {
            "name": "content_constrained",
            "description": "Any question (natural or search) that specifies inclusions/exclusions or a particular perspective (e.g., ‘for developers’, ‘excluding finance’).",
            "probability": 0.05
        },
        {
            "name": "advanced_reasoning",
            "description": "Complex, multi-step questions requiring deep analysis, multi-source cross-referencing, or multi-hop inference (e.g., “Evaluate how policy changes in X region affected economic indicators over the past decade”).",
            "probability": 0.05
        },
        {
            "name": "sensitive",
            "description": "Questions involving personal privacy, harmful instructions, or controversial topics.",
            "probability": 0.02
        }
    ]
}

answer_type = {
    "categorization_name": "answer_type",
    "categories": [
        {
            "name": "factoid",
            "description": "a question seeking a specific, concise piece of information or a short fact about a particular subject, such as a name, date, or number.",
            "probability": 0.1,
            "is_multi_doc": False
        },
        {
            "name": "procedural",
            "description": "“How-to” questions asking for step-by-step instructions or processes.",
            "probability": 0.1,
            "is_multi_doc": False
        },
        {
            "name": "multi-aspect",
            "description": ("A question about two different aspects of the same entity/concept. "
                            "For example: 'What are the advantages of AI-powered diagnostics, and what are the associated risks of bias in medical decision-making?', "
                            "'How do cryptocurrencies enable financial inclusion, and what are the security risks associated with them?'. "
                            "The information required to answer the question needs to come from two documents, "
                            "specifically, the first document must provide information about the first aspect, while the second must provide information about the second aspect."),
            "probability": 0.3,
            "is_multi_doc": True
        },
        {
            "name": "comparison",
            "description": ("a comparison question that requires comparing two related concepts or entities. "
                            "The comparison must be natural and reasonable, i.e., comparing two entities by a common attribute which is meaningful and relevant to both entities. "
                            "For example: 'Who is older, Glenn Hughes or Ross Lynch?', 'Are Pizhou and Jiujiang in the same province?', "
                            "'Pyotr Ilyich Tchaikovsky and Giuseppe Verdi have this profession in common'. "
                            "The information required to answer the question needs to come from two documents, specifically, "
                            "the first document must provide information about the first entity/concept, while the second must provide information about the second entity/concept."),
            "probability": 0.5,
            "is_multi_doc": True
        }
    ]
}

temporal_focus = {
    "categorization_name": "temporal_focus",
    "categories": [
        {
            "name": "historical",
            "description": "The question refers to events, developments, or information rooted in the past—common in news archives, historical analysis, or retrospectives.",
            "probability": 0.3
        },
        {
            "name": "current",
            "description": "The question focuses on recent or ongoing events, trends, or updates—frequent in news, blogs, or product announcements.",
            "probability": 0.4
        },
        {
            "name": "future",
            "description": "The question is oriented toward future events, forecasts, innovations, or hypothetical scenarios—common in technology blogs, academic projections, or policy discussions.",
            "probability": 0.1
        },
        {
            "name": "atemporal",
            "description": "The question concerns general knowledge, definitions, or conceptual content not tied to a specific time frame—typical of encyclopedic entries, tutorials, or evergreen content.",
            "probability": 0.2
        }
    ]
}

all_question_categorizations_except_answer_type = [
    premise_categorization,
    multilingual_categorization,
    question_length,
    answer_length,
    question_formulation,
    temporal_focus
]


def select_random_question_categorizations():
    # Always include answer_type
    selected = [answer_type]
    # Randomly select 3 other categorizations
    selected += random.sample(all_question_categorizations_except_answer_type, 4)
    return selected


# Example usage
# pprint.pprint(select_random_question_categorizations())
