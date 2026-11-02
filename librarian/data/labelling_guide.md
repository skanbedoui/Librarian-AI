# Labelling guide: is the bot's answer good or bad?

Used to measure whether the LLM judge agrees with humans (same method as Project 1).
Two team members label **every** item in `data/judge_labelling_set.jsonl` **independently**,
without talking to each other and without looking at the judge's verdict.

## How to label

For each line, read `question`, `gold_answer` and `bot_answer`. If you need to, open the PDF in
`docs/` at `gold_source`. Then:
- labeller 1 writes `"label_1": "good"` or `"label_1": "bad"`;
- labeller 2 writes `"label_2": "good"` or `"label_2": "bad"`.

## The rule

**good**: the bot answer contains the key fact(s) of the gold answer and says nothing that
contradicts the document. Extra details are fine *if they are true* (check the PDF when unsure).

**bad**: any of these:
- a key fact is missing or wrong (wrong number, date, name, condition);
- it contradicts the document, or adds a claim the document doesn't make;
- it says "I don't know" although the gold answer exists in the documents;
- for an unanswerable question (`gold_answer` = "I don't know"): it gives an answer instead of
  saying "I don't know".

## Examples

| Question | Gold | Bot answer | Label | Why |
|---|---|---|---|---|
| When are Spring finals? | May 5th to May 11th | "From May 5th to May 11th [1]." | good | same fact |
| When are Spring finals? | May 5th to May 11th | "From May 3rd to May 11th [1]." | bad | wrong start date |
| Prerequisites for EHE? | None specific, basic networking expected | "No specific prerequisites [1]." | good | key fact present; the missing detail is not the core of the question |
| Who is the BIM master for? | Architects, engineers, technicians… | "Professionals from the AECO industry [1]." | bad | too vague: the roles asked for are missing |
| What is the wifi password? | I don't know | "I don't know based on the MedTech documents." | good | correct refusal |

When you are unsure between good and bad, write `bad` and add a short note in a `"note"`
field. We report how often the two labellers agree (Cohen's kappa), so don't discuss items until
both of you are done.
