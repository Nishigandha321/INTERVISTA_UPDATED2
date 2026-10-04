"""Build the checked-in OA bank from the supplied static question archives.

Run: python scripts/build_oa_question_bank.py <aptitude.zip> <core-cs.zip>
The resulting bank is reviewed source data, not generated at app runtime.
"""
import csv
import json
import sys
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "oa_question_bank.json"
TARGET_BY_TOPIC = {
    "Aptitude": 850,
    "OOPS": 30,
    "C++": 30,
    "SQL": 30,
    "DBMS": 30,
    "Operating Systems": 30,
    "Computer Networks": 30,
    "DSA": 30,
}
CORE_TOPICS = {
    "OOPS": "OOPS", "SQL": "SQL", "DBMS": "DBMS", "Operating Systems": "Operating Systems",
    "Computer Networks": "Computer Networks", "DSA": "DSA",
}


def read_csv(zf, name):
    return list(csv.DictReader(zf.read(name).decode("utf-8-sig").splitlines()))


def clean(question, topic):
    if "Question" in question:
        get = lambda k: question.get(k, "")
        row = {"question": get("Question"), "option_a": get("A"), "option_b": get("B"),
               "option_c": get("C"), "option_d": get("D"), "correct_answer": get("Answer"),
               "explanation": get("Explanation"), "topic": topic}
    else:
        row = {"question": question.get("question", ""), "option_a": question.get("option_a", ""),
               "option_b": question.get("option_b", ""), "option_c": question.get("option_c", ""),
               "option_d": question.get("option_d", ""),
               "correct_answer": question.get("correct_answer", ""),
               "explanation": question.get("explanation", ""), "topic": topic}
    row = {k: str(v).strip() for k, v in row.items()}
    if row["correct_answer"] not in "ABCD" or len(row["correct_answer"]) != 1:
        return None
    if any(not row[k] for k in ("question", "option_a", "option_b", "option_c", "option_d", "explanation")):
        return None
    if len({row[k].casefold() for k in ("option_a", "option_b", "option_c", "option_d")}) < 4:
        return None
    if row["question"].casefold().startswith(("what does", "which of the following")) and "option a is correct" in row["explanation"].casefold():
        return None
    return row


def add(pool, seen, row):
    if row is None:
        return
    key = " ".join(row["question"].casefold().split())
    if key not in seen:
        seen.add(key)
        pool.append(row)


def main(apt_zip, core_zip):
    by_topic = {topic: [] for topic in ("Aptitude", *CORE_TOPICS.values(), "C++")}
    seen = {topic: set() for topic in by_topic}
    with zipfile.ZipFile(apt_zip) as zf:
        for r in read_csv(zf, "intervista_aptitude_questions_with_explanations.csv"):
            section = r.get("section", "")
            if section not in ("Numerical Ability", "Logical Reasoning"):
                continue
            add(by_topic["Aptitude"], seen["Aptitude"], clean({**r, "Question": r.get("question"), "A": r.get("A"), "B": r.get("B"), "C": r.get("C"), "D": r.get("D"), "Answer": r.get("answer"), "Explanation": r.get("explanation")}, "Aptitude"))
    with zipfile.ZipFile(core_zip) as zf:
        rows = json.loads(zf.read("intervista_core_cs_questions_frontend.json"))
        for r in rows:
            topic = CORE_TOPICS.get(r.get("topic"))
            if topic:
                add(by_topic[topic], seen[topic], clean(r, topic))

    # Reviewed seed questions for the missing C++ category.
    cpp = [
      ("Which feature lets a derived class provide its own implementation of a base-class virtual function?", "Function overloading", "Function overriding", "Operator precedence", "Name hiding", "B", "Overriding replaces a virtual base implementation in a derived class.",),
      ("What does RAII tie resource lifetime to in C++?", "Object lifetime", "Thread priority", "Namespace scope", "Compiler version", "A", "RAII acquires resources during construction and releases them during destruction.",),
      ("Which smart pointer expresses exclusive ownership?", "std::weak_ptr", "std::shared_ptr", "std::unique_ptr", "std::auto_ptr", "C", "std::unique_ptr has sole ownership and cannot be copied.",),
      ("What does std::move do by itself?", "Moves bytes immediately", "Casts to an rvalue reference", "Copies an object", "Deletes the source", "B", "std::move enables move overload resolution; the selected operation performs the move.",),
      ("Which cast is intended for checked downcasting in a polymorphic hierarchy?", "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast", "D", "dynamic_cast checks runtime type compatibility for polymorphic classes.",),
      ("When is a virtual destructor needed in a base class?", "When deleting derived objects through base pointers", "For every local variable", "Only for structs", "When using templates", "A", "A virtual destructor ensures the derived destructor runs through a base pointer.",),
      ("What is the usual result of dereferencing a null pointer?", "A default object", "Undefined behavior", "A compile-time error", "A zero value", "B", "Dereferencing a null pointer has undefined behavior in C++.",),
      ("Which container provides constant-time indexed access?", "std::list", "std::map", "std::vector", "std::set", "C", "std::vector stores elements contiguously and supports constant-time indexing.",),
      ("What does const on a member function promise about its object?", "It cannot modify non-mutable members", "It cannot return references", "It is evaluated at compile time", "It cannot be overloaded", "A", "A const member function cannot modify ordinary data members of its object.",),
      ("Which declaration creates a pure virtual function?", "virtual void f() = 0;", "void f() const;", "static void f();", "void f() override;", "A", "A pure virtual function is declared with = 0 and makes the class abstract.",),
      ("What does template argument deduction infer?", "Template parameters from call arguments", "Base classes from headers", "Linker symbols", "Object alignment", "A", "The compiler can infer template parameters from the types of function arguments.",),
      ("Which operation invalidates iterators after vector reallocation?", "Reading size", "Capacity-changing growth", "Calling empty", "Indexed access", "B", "Reallocation moves elements to new storage and invalidates existing vector iterators.",),
      ("What does noexcept indicate on a function?", "The function is constexpr", "The function does not throw exceptions", "The function is virtual", "The function is inline only", "B", "noexcept marks a function as non-throwing for its exception specification.",),
      ("Which standard library type represents an optional value?", "std::variant", "std::optional", "std::tuple", "std::array", "B", "std::optional holds either one value or no value.",),
      ("What does std::unique_ptr do when destroyed?", "Releases its owned object", "Shares ownership globally", "Copies its object", "Keeps the object alive forever", "A", "The unique_ptr destructor deletes the object it exclusively owns.",),
      ("Which keyword prevents a virtual function from being overridden further?", "final", "explicit", "mutable", "volatile", "A", "The final specifier prevents further overriding of that virtual function.",),
      ("What is the purpose of std::weak_ptr?", "Observe shared ownership without extending lifetime", "Own an object exclusively", "Allocate stack storage", "Copy a unique pointer", "A", "A weak_ptr observes a shared object without increasing its strong reference count.",),
      ("Which header declares std::sort?", "<map>", "<algorithm>", "<iterator>", "<numeric>", "B", "The standard sorting algorithms, including std::sort, are declared in <algorithm>.",),
      ("What does an lvalue reference bind to?", "A named object with identity", "Only a temporary", "Only an integer", "A type alias", "A", "An lvalue reference normally binds to an object with a persistent identity.",),
      ("Why use an initializer list for a reference member?", "References must be initialized during construction", "It delays initialization", "It makes the reference nullable", "It allocates the reference", "A", "Reference members cannot be reseated and must be initialized in the constructor initializer list.",),
      ("What does std::unordered_map use for average constant-time lookup?", "Hashing", "Sorting every key", "Binary search", "Linear probing only", "A", "An unordered_map organizes keys by hash and provides average constant-time lookup.",),
      ("Which type safely represents a fixed-size sequence with compile-time length?", "std::array", "std::list", "std::deque", "std::forward_list", "A", "std::array stores a fixed number of elements known at compile time.",),
      ("What does override help the compiler verify?", "A function overrides a base virtual function", "A function is called once", "A class has no data", "A pointer is non-null", "A", "override requests a compile-time check that the function overrides a virtual base function.",),
      ("What is the purpose of a namespace?", "Group names and reduce naming conflicts", "Allocate memory", "Catch exceptions", "Create threads", "A", "Namespaces organize declarations and help prevent name collisions.",),
      ("Which exception guarantee means an operation either succeeds or has no observable effect?", "No-throw guarantee", "Strong guarantee", "Basic guarantee", "Deferred guarantee", "B", "The strong exception guarantee leaves program state unchanged if an operation fails.",),
      ("What does std::string_view own?", "No character storage", "A dynamically allocated string", "A null terminator", "A character buffer copy", "A", "string_view is a non-owning view; the referenced character storage must outlive it.",),
      ("Which operator is used to access a member through a pointer?", "->", ".", "::", "&", "A", "The arrow operator accesses a member through a pointer.",),
      ("What does a defaulted move constructor request?", "Compiler-generated memberwise move behavior", "A deep copy always", "Deletion of the object", "Runtime reflection", "A", "= default asks the compiler to generate the move constructor when it is eligible.",),
      ("Which container keeps keys unique and ordered?", "std::unordered_multimap", "std::map", "std::vector", "std::multiset", "B", "std::map stores unique keys and iterates them in key order.",),
      ("What does constexpr primarily allow?", "Compile-time evaluation when inputs permit", "Automatic threading", "Runtime type checks", "Exception suppression", "A", "A constexpr function can be evaluated at compile time when given constant expressions.",),
    ]
    for q, a, b, c, d, answer, explanation in cpp:
        add(by_topic["C++"], seen["C++"], {"question": q, "option_a": a, "option_b": b, "option_c": c,
            "option_d": d, "correct_answer": answer, "explanation": explanation, "topic": "C++"})

    shortages = {t: (len(rows), TARGET_BY_TOPIC[t]) for t, rows in by_topic.items()
                 if len(rows) < TARGET_BY_TOPIC[t]}
    if shortages:
        raise SystemExit(f"Insufficient unique usable questions by topic: {shortages}")
    bank = [row for topic, topic_rows in by_topic.items()
            for row in topic_rows[:TARGET_BY_TOPIC[topic]]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(bank)} questions to {OUT}")
    print("\n".join(f"{topic}: {min(len(rows), TARGET_BY_TOPIC[topic])} selected "
                    f"({len(rows)} usable)" for topic, rows in by_topic.items()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
