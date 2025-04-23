import random

user_follow_up_likelihood = {
    "categorization_name": "user_follow_up_likelihood",
    "categories": [
        {
            "name": "likely_follow_up",
            "description": "An open-ended or exploratory question—often seen in blog discussions or academic prompts—that naturally invites further clarification or deeper inquiry.",
            "probability": 0.35
        },
        {
            "name": "moderate_follow_up",
            "description": "A moderately scoped question—common in how-to guides or analytical queries—that could generate follow-ups but stands reasonably self-contained.",
            "probability": 0.40
        },
        {
            "name": "unlikely_follow_up",
            "description": "A tightly scoped, specific question—typical of FAQs or product-spec queries—unlikely to prompt additional questions.",
            "probability": 0.25
        }
    ]
}

user_language_proficiency = {
    "categorization_name": "user_language_proficiency",
    "categories": [
        {
            "name": "english_native_fluent",
            "description": "Writes idiomatically in English with few or no grammatical errors.",
            "probability": 0.60
        },
        {
            "name": "spanish_native_fluent",
            "description": "Writes idiomatically in Spanish with few or no grammatical errors.",
            "probability": 0.15
        },
        {
            "name": "chinese_native_fluent",
            "description": "Writes idiomatically in Chinese with few or no grammatical errors.",
            "probability": 0.10
        },
        {
            "name": "non_native_learner",
            "description": "Shows signs of language learning—simpler structures, occasional grammar errors, or direct translations—in any language.",
            "probability": 0.15
        }
    ]
}

user_expertise = {
    "categorization_name": "user_profile",
    "categories": [
        {
            "name": "expert_specialist",
            "description": "High domain expertise (uses technical terminology correctly), asks highly specific questions with clear constraints and desired formats.",
            "probability": 0.40
        },
        {
            "name": "knowledgeable_generalist",
            "description": "Good familiarity with the topic (aware of key concepts), asks moderately specific questions—providing some context but leaving room for interpretation.",
            "probability": 0.30
        },
        {
            "name": "novice_inquirer",
            "description": "Limited domain familiarity, asks basic or broad questions with vague or ambiguous phrasing and few constraints.",
            "probability": 0.30
        }
    ]
}

user_preferred_response_style = {
    "categorization_name": "user_preferred_response_style",
    "categories": [
        {
            "name": "concise_direct",
            "description": "User asks very direct, keyword-like questions implying a quick fact or definition is desired.",
            "probability": 0.25
        },
        {
            "name": "conceptual_explanation",
            "description": "User asks “why” or “how,” focuses on understanding underlying concepts or theory.",
            "probability": 0.30
        },
        {
            "name": "practical_steps",
            "description": "User requests “how to” instructions or step-by-step processes.",
            "probability": 0.15
        },
        {
            "name": "illustrative_examples",
            "description": "User asks for examples, use cases, or “such as” to clarify a concept.",
            "probability": 0.15
        },
        {
            "name": "structured_comparison",
            "description": "User implicitly or explicitly wants structured output—tables, pros/cons, or direct comparisons between items.",
            "probability": 0.15
        }
    ]
}

user_intent = {
    "categorization_name": "user_intent",
    "categories": [
        {
            "name": "information_seeking",
            "description": "User wants factual or conceptual knowledge, whether for academic, professional, personal, or practical purposes.",
            "probability": 0.70
        },
        {
            "name": "instructional",
            "description": "User asks the system to perform a transformation or generation task (e.g., summarize, translate, write code, generate a list).",
            "probability": 0.10
        },
        {
            "name": "comparison",
            "description": "User seeks to compare similarities or differences between two or more entities or concepts.",
            "probability": 0.10
        },
        {
            "name": "opinion_recommendation",
            "description": "User requests subjective viewpoints, evaluations, or recommendations.",
            "probability": 0.10
        }
    ]
}

user_conversational_style = {
    "categorization_name": "user_conversational_style",
    "categories": [
        {
            "name": "formal_transactional",
            "description": "Uses formal, grammatically complete language without slang or filler; focuses directly on the query (e.g., business reports, academic questions).",
            "probability": 0.20
        },
        {
            "name": "polite_conversational",
            "description": "Uses neutral-to-polite phrasing (e.g., ‘please’, greetings), complete sentences, and a friendly but focused tone.",
            "probability": 0.40
        },
        {
            "name": "informal_direct",
            "description": "Employs casual language, slang or abbreviations, sentence fragments, and issues commands or urgent requests without pleasantries.",
            "probability": 0.20
        },
        {
            "name": "informal_expressive",
            "description": "Casual phrasing combined with emotional indicators—enthusiasm, frustration, emojis, or exclamation—often seen in blog comments or social media excerpts.",
            "probability": 0.20
        }
    ]
}

all_user_categories = [
    user_follow_up_likelihood,
    user_expertise,
    user_preferred_response_style,
    user_intent,
    user_conversational_style
]


def select_random_user_categories():
    # Always include answer_type
    selected = [user_language_proficiency]
    # Randomly select 3 other categorizations
    selected += random.sample(all_user_categories, 3)
    return selected


# Example usage:
# print(select_random_user_categories())

