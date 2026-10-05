"""Server-only test drivers for function-based Coding Round questions."""

from __future__ import annotations

import re
from typing import Any


CPP_PREFIX = "#include <bits/stdc++.h>\nusing namespace std;\n"
CPP_VECTOR_PRINTER = r"""
template <typename T>
static void printVector(const vector<T>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) cout << ' ';
        cout << values[i];
    }
    cout << '\n';
}
"""

CPP_DRIVERS = {
    1: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;int t;cin>>t;printVector(__FUNCTION__(v,t));}",
    2: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;cout<<__FUNCTION__(v)<<'\\n';}",
    3: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;cout<<__FUNCTION__(v)<<'\\n';}",
    4: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;printVector(__FUNCTION__(v));}",
    5: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;printVector(__FUNCTION__(v));}",
    6: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;cout<<__FUNCTION__(v)<<'\\n';}",
    7: "int main(){string s;cin>>s;cout<<__FUNCTION__(s)<<'\\n';}",
    8: "int main(){string s;cin>>s;cout<<(__FUNCTION__(s)?\"YES\":\"NO\")<<'\\n';}",
    9: "int main(){string s,t;getline(cin,s);getline(cin,t);cout<<(__FUNCTION__(s,t)?\"YES\":\"NO\")<<'\\n';}",
    10: "int main(){string s;cin>>s;cout<<__FUNCTION__(s)<<'\\n';}",
    11: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;int t;cin>>t;cout<<__FUNCTION__(v,t)<<'\\n';}",
    12: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;int t;cin>>t;cout<<__FUNCTION__(v,t)<<'\\n';}",
    13: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;printVector(__FUNCTION__(v));}",
    14: "int main(){int n;cin>>n;vector<int>a(n);for(int&x:a)cin>>x;int m;cin>>m;vector<int>b(m);for(int&x:b)cin>>x;printVector(__FUNCTION__(a,b));}",
    15: "int main(){int n;cin>>n;vector<int> v(n);for(int&x:v)cin>>x;printVector(__FUNCTION__(v));}",
    16: "int main(){string s;cin>>s;cout<<(__FUNCTION__(s)?\"YES\":\"NO\")<<'\\n';}",
    17: "int main(){int q;cin>>q;cin.ignore(numeric_limits<streamsize>::max(),'\\n');vector<string> ops(q);for(string&op:ops)getline(cin,op);printVector(__FUNCTION__(ops));}",
    18: "int main(){int n;cin>>n;cout<<__FUNCTION__(n)<<'\\n';}",
    19: "int main(){int n;cin>>n;cout<<__FUNCTION__(n)<<'\\n';}",
    20: "int main(){long long a,b;cin>>a>>b;cout<<__FUNCTION__(a,b)<<'\\n';}",
}

PYTHON_DRIVERS = {
    1: "data=list(map(int,sys.stdin.read().split()));n=data[0];v=data[1:n+1];t=data[n+1];print(*__FUNCTION__(v,t))",
    2: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(__FUNCTION__(data[1:n+1]))",
    3: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(__FUNCTION__(data[1:n+1]))",
    4: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(*__FUNCTION__(data[1:n+1]))",
    5: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(*__FUNCTION__(data[1:n+1]))",
    6: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(__FUNCTION__(data[1:n+1]))",
    7: "print(__FUNCTION__(sys.stdin.read().strip()))",
    8: "print('YES' if __FUNCTION__(sys.stdin.read().strip()) else 'NO')",
    9: "s,t=sys.stdin.read().splitlines();print('YES' if __FUNCTION__(s,t) else 'NO')",
    10: "print(__FUNCTION__(sys.stdin.read().strip()))",
    11: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(__FUNCTION__(data[1:n+1],data[n+1]))",
    12: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(__FUNCTION__(data[1:n+1],data[n+1]))",
    13: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(*__FUNCTION__(data[1:n+1]))",
    14: "data=list(map(int,sys.stdin.read().split()));n=data[0];a=data[1:n+1];m=data[n+1];b=data[n+2:n+2+m];print(*__FUNCTION__(a,b))",
    15: "data=list(map(int,sys.stdin.read().split()));n=data[0];print(*__FUNCTION__(data[1:n+1]))",
    16: "print('YES' if __FUNCTION__(sys.stdin.read().strip()) else 'NO')",
    17: "lines=sys.stdin.read().splitlines();q=int(lines[0]);result=__FUNCTION__(lines[1:1+q]);print('\\n'.join(map(str,result)))",
    18: "print(__FUNCTION__(int(sys.stdin.read().strip())))",
    19: "print(__FUNCTION__(int(sys.stdin.read().strip())))",
    20: "a,b=map(int,sys.stdin.read().split());print(__FUNCTION__(a,b))",
}


def validate_candidate_function(question: dict[str, Any], language: str, source_code: str) -> None:
    """Require the question's function and reject candidate-owned entry points."""
    name = re.escape(str(question.get("function_name", "")))
    if not name:
        raise ValueError("This coding question has no function signature configured.")
    if language == "cpp":
        has_function = re.search(rf"\b{name}\s*\(", source_code) is not None
    else:
        has_function = re.search(rf"\bdef\s+{name}\s*\(", source_code) is not None
    if not has_function:
        raise ValueError(f"Implement the provided {question['function_name']} function signature.")
    if has_entrypoint(language, source_code):
        raise ValueError("Submit only the provided function. The test driver is supplied by Intervista.")


def has_entrypoint(language: str, source_code: str) -> bool:
    """Identify legacy/full-program submissions so they are never restored in the editor."""
    if language == "cpp":
        return re.search(r"\bmain\s*\(", source_code) is not None
    return re.search(r"\bdef\s+main\s*\(|__name__\s*==", source_code) is not None


def build_execution_source(question: dict[str, Any], language: str, source_code: str) -> str:
    """Join candidate function code to a private input parser and output driver."""
    validate_candidate_function(question, language, source_code)
    question_id = int(question["id"])
    function_name = str(question["function_name"])
    if language == "cpp":
        driver = CPP_DRIVERS.get(question_id)
        if driver is None:
            raise ValueError("No C++ test driver is configured for this question.")
        return CPP_PREFIX + source_code + "\n" + CPP_VECTOR_PRINTER + driver.replace("__FUNCTION__", function_name) + "\n"
    if language == "python":
        driver = PYTHON_DRIVERS.get(question_id)
        if driver is None:
            raise ValueError("No Python test driver is configured for this question.")
        return "import sys\n" + source_code + "\n\n" + driver.replace("__FUNCTION__", function_name) + "\n"
    raise ValueError("Choose C++ or Python.")
