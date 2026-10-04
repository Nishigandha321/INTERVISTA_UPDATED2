# Static OA question bank

`oa_question_bank.json` is the checked-in runtime source for OA questions. It contains 1,060 records: 850 Aptitude questions and 30 each for OOPS, C++, SQL, DBMS, Operating Systems, Computer Networks, and DSA. Each OA samples the same 10-topic mix configured in `main.py`; answer keys and explanations stay in the server session, and only question text and options are sent to the browser.

The bank builder, `scripts/build_oa_question_bank.py`, takes the supplied aptitude and core CS archives as input. It maps Numerical Ability and Logical Reasoning to Aptitude, selects the supplied core CS questions, removes duplicate question text and malformed records, and includes 30 reviewed C++ seed questions because the supplied core archive has no C++ category. The aptitude archive contributes 850 of 1,237 unique usable questions. The resulting JSON is committed so runtime operation does not depend on the source ZIPs or an LLM.

To rebuild it, run:

```text
python scripts/build_oa_question_bank.py <aptitude.zip> <core-cs.zip>
```

The bank holds the question text, four options, answer letter, explanation, and OA topic. Edit the JSON when curating the live bank; retain the exact keys and topic names expected by `main.py`.
