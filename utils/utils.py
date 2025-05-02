import json

data = [
    {'doc_id': '<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>', 'text': "What will the age of Aquarius be like. What is second level consciousness and are we still there? Left and Right Brain technology what is Left and Right Brain technology? part 1\nSorry we couldn't complete your registration. Please try again. You must accept the Terms and conditions to register.", 'score': 13.5859375, 'source': 'dense', 'uid': '3e6c9e47-fa3e-4408-a773-21a247e99066', 'mxbai_score': 0.578305184841156},
    {'doc_id': '<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>', 'text': 'Title: Left-Brain/Right-Brain Functions\nPreview: We have two eyes, two ears, two hands, and two minds. Our left brain is the side used more by writiers, mathematicians, and scientists; the right side by artists, craftspeople, and musicians. By aterry (adrienne)\non September 27, 2012.', 'score': -2.0390625, 'source': 'dense', 'uid': 'e79fa322-fd73-42c9-852f-bc25dd665f34', 'mxbai_score': 0.35160550475120544},
    {'doc_id': '<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>', 'text': "Our left brain thinks in terms of words and symbols while our right brain thinks in terms of images. Remembering a persons name is a function of the left-brain memory while rembering a persons's face is a function.", 'score': -4.41796875, 'source': 'dense', 'uid': 'f7d06346-31d7-4635-8647-daaa01670e60', 'mxbai_score': 0.3865794837474823},
    {'doc_id': '<urn:uuid:e4bf2415-2032-4a8a-9c18-715cf2d5f91f>', 'text': 'The legal term for this compensation is “damages. ” Exactly what damages you can recover varies from state to state, but you can usually recover:\n- Past and future medical expenses\n- Future lost wages (if the injury limits your ability to work in the future)\n- Property damages\n- Pain and suffering\n- Emotional distress\nReady to contact a lawyer about a possible second impact syndrome case? Use our free online directory to schedule your initial consultation today. - Guide to traumatic brain injuries\n- Resources to help after a brain injury\n- How to recognize a brain injury and what you should do about a brain injury\n- Concussions and auto accidents\n- Rehabilitation and therapy after a brain injury\n- Second impact syndrome and sports injury lawsuits\n- Legal guide to brain death\n- What is CTE?\n- A loss of oxygen can lead to an anoxic brain injury\n- Can you recover costs for the accident that caused a brain bleed?\n- What is the Traumatic Brain Injury Act?\n- Understanding the Hidden Challenges of Mild Traumatic Brain Injury\n- What is the Glasgow Coma Scale?.', 'score': -17.671875, 'source': 'sparse', 'uid': '466ac00f-ba30-4ca6-b8ff-d95a0e32f1a8', 'mxbai_score': 0.29968908429145813}
]

test_question_path = "../data/question/current_questions.jsonl"
test_answer_path = "../data/answer/current_answers.jsonl"


def merge_scores_and_keep_positive(entries):
    for entry in entries:
        if 'mxbai_score' in entry:
            entry['score'] += entry['mxbai_score']
            del entry['mxbai_score']
        if 'uid' in entry:
            del entry['uid']
    # Sort the entries by the updated 'score' in descending order
    entries.sort(key=lambda x: x['score'], reverse=True)
    return [doc for doc in entries if doc.get('score', 0) > 0]


def create_test_question():
    # Load the JSONL file
    file_path = "../data/generated/results_id_0facd9cc-06f4-4922-8c1f-e29811d5a08c_user_id_bfc67a4c-41ca-4a55-87c0-85fe0e346a90.jsonl"
    processed_data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file, start=1):
            item = json.loads(line)
            processed_item = {
                "id": idx,
                "question": item.get("question"),
                "answer_ground": item.get("answer"),
                "context": item.get("context"),
                "document_ids": item.get("document_ids")
            }
            processed_data.append(processed_item)
    # Save the modified data to a new JSONL file
    output_path = test_question_path
    with open(output_path, "w", encoding="utf-8") as file:
        for item in processed_data:
            file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    # Output the modified array
    # print(merge_scores_and_clean(data))
    create_test_question()
