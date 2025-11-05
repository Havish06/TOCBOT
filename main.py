import os
import re
import json
from collections import deque
from typing import Dict, List, Any, Tuple
from datetime import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

try:
    from flask import Flask, request, jsonify, render_template_string
    FLASK_AVAILABLE = True
except Exception:
    FLASK_AVAILABLE = False

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


# ===================== PERPLEXITY CLIENT =====================
class PerplexityClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar"

    def is_available(self):
        return bool(self.api_key) and REQUESTS_AVAILABLE

    def chat(self, message: str, history: List[Dict] = None, timeout: int = 30) -> str:
        if not self.is_available():
            return "❌ Perplexity API not available or key missing."

        try:
            msgs = []
            if history:
                for h in history[-8:]:
                    role = "user" if h.get("role") == "user" else "assistant"
                    content = h.get("message", "")
                    if content:
                        msgs.append({"role": role, "content": content})
            msgs.append({"role": "user", "content": message})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system",
                     "content": "You are a chill, friendly AI. Keep it short and natural. No citations."}
                ] + msgs,
                "temperature": 0.9,
                "disable_search": True,
                "return_related_sources": False,
                "search_domain_filter": ["chat"],
            }



            resp = requests.post(self.base_url, json=payload, headers=headers, timeout=timeout)
            print(f"DEBUG: Perplexity status {resp.status_code}")
            print(f"DEBUG: Response text: {resp.text[:300]}")

            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                else:
                    return "⚠️ No message content in API response."
            else:
                return f"❌ API error {resp.status_code}: {resp.text[:200]}"

        except Exception as e:
            import traceback
            print("DEBUG: Exception:", traceback.format_exc())
            return f"⚠️ Exception while calling Perplexity: {e}"


# ===================== SPECIALIZED ENGINE =====================
class SpecializedEngine:
    def __init__(self):
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

    def evaluate_math(self, expr):
        try:
            expr_mod = expr.replace('^', '**')
            if SYMPY_AVAILABLE:
                return f"Result: {sp.N(sp.sympify(expr_mod))}"
            result = eval(expr_mod, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Math error: {e}"

    def parse_sentence(self, sentence):
        s = sentence.strip()
        if not s:
            return "✗ Empty sentence."
        if self.nlp:
            doc = self.nlp(s)
            deps = [t.dep_.lower() for t in doc]
            pos = [t.pos_ for t in doc]
            has_verb = any(p == "VERB" or p == "AUX" for p in pos)
            has_subject = any(d in ("nsubj", "nsubjpass", "csubj", "expl") for d in deps)
            if has_subject and has_verb:
                return "✓ Valid English sentence."
            if has_verb:
                return "✓ Likely valid sentence."
            return "✗ Grammatically incorrect."
        tokens = s.split()
        if len(tokens) < 3:
            return "✗ Too short to be valid."
        if re.search(r"\b(am|is|are|was|were|be|been|being|do|does|did|have|has|had|will|would|shall|should|can|could|may|might|must)\b", s.lower()):
            return "✓ Probably valid English sentence."
        return "✗ Seems incomplete or incorrect."

    def check_dfa_ends_01(self, binary):
        if not re.fullmatch(r"[01]+", binary):
            return "Invalid input: use only 0 and 1"
        return "✓ Accepted" if binary.endswith("01") else "✗ Rejected"

    def check_pda_balanced(self, expr):
        stack = []
        for ch in expr:
            if ch == '(':
                stack.append(ch)
            elif ch == ')':
                if not stack:
                    return "✗ Unbalanced"
                stack.pop()
        return "✓ Balanced" if not stack else "✗ Unbalanced"

    def test_regex(self, pattern, text):
        try:
            return "✓ Match" if re.fullmatch(pattern, text) else "✗ No match"
        except re.error as e:
            return f"Regex error: {e}"


# ===================== INTENT CLASSIFIER =====================
class IntentClassifier:
    @staticmethod
    def classify(text):
        t = text.strip()
        l = t.lower()
        if l == "help":
            return "help", {}
        if l == "clear":
            return "command", {"cmd": "clear"}
        if "daily conversation" in l or "everyday chat" in l:
            return "daily", {"message": t}
        if l.startswith("parse:"):
            return "parse", {"sentence": t.split(":", 1)[1].strip()}
        if l.startswith("dfa:"):
            return "dfa", {"input": t.split(":", 1)[1].strip()}
        if l.startswith("pda:"):
            return "pda", {"input": t.split(":", 1)[1].strip()}
        if l.startswith("regex:"):
            try:
                tail = t.split(":", 1)[1]
                parts = tail.split(";")
                pattern = parts[0].strip()
                rest = ";".join(parts[1:])
                m = re.search(r"string\s*:\s*(.*)\Z", rest, flags=re.IGNORECASE | re.DOTALL)
                s = m.group(1).strip() if m else ""
                return "regex", {"pattern": pattern, "string": s}
            except Exception:
                return "general", {}
        if re.match(r"^[\d\+\-\*/\(\)\.\^ %]+$", l):
            return "math", {"expression": t}
        return "general", {}


# ===================== DIALOGUE + BOT =====================
class DialogueManager:
    def __init__(self):
        self.history = deque(maxlen=30)

    def add(self, role, message):
        self.history.append({"role": role, "message": message, "timestamp": datetime.now().isoformat()})

    def get_history(self):
        return list(self.history)

    def clear(self):
        self.history.clear()


class HybridChatbot:
    def __init__(self, api_key=None):
        self.perplexity = PerplexityClient(api_key)
        self.engine = SpecializedEngine()
        self.classifier = IntentClassifier()
        self.dialogue = DialogueManager()

    def chat(self, text):
        intent, params = self.classifier.classify(text)
        if intent == "command" and params.get("cmd") == "clear":
            self.dialogue.clear()
            return {"response": "✓ Conversation cleared"}
        if intent == "help":
            return {"response": self._help_text()}
        if intent == "daily":
            if self.perplexity.is_available():
                reply = self.perplexity.chat(params.get("message", text), self.dialogue.get_history())
                if not reply:
                    reply = "External API didn’t respond."
            else:
                reply = "⚠️ Perplexity API not configured. Set PERPLEXITY_API_KEY."
            self.dialogue.add("user", text)
            self.dialogue.add("assistant", reply)
            return {"response": reply}
        if intent == "parse":
            reply = self.engine.parse_sentence(params.get("sentence", text))
        elif intent == "dfa":
            reply = self.engine.check_dfa_ends_01(params.get("input", text))
        elif intent == "pda":
            reply = self.engine.check_pda_balanced(params.get("input", text))
        elif intent == "regex":
            reply = self.engine.test_regex(params.get("pattern", ""), params.get("string", ""))
        elif intent == "math":
            reply = self.engine.evaluate_math(params.get("expression", text))
        else:
            reply = self._fallback_text()
        self.dialogue.add("user", text)
        self.dialogue.add("assistant", reply)
        return {"response": reply}

    def _help_text(self):
        return (
            "Help\n\n"
            "Parsing\n"
            "• parse: <sentence> — returns ✓ Valid or ✗ Incorrect.\n\n"
            "Theory of Computation\n"
            "• dfa: <binary> — accepts if it ends with 01.\n"
            "• pda: <expr> — checks balanced parentheses.\n"
            "• regex: <pattern>; string: <text> — full-match test.\n"
            "• math: <expression> — evaluate safely.\n\n"
            "Daily Chat\n"
            "• Include 'daily conversation' or 'everyday chat' to use Perplexity.\n\n"
            "Other\n"
            "• help — show this\n"
            "• clear — reset conversation"
        )

    def _fallback_text(self):
        return (
            "Local Tools:\n"
            "• parse: <sentence>\n"
            "• dfa: <binary>\n"
            "• pda: <parentheses>\n"
            "• regex: <pattern>; string: <text>\n"
            "• math: e.g., 2^10 + 5\n\n"
            "For normal convo, include 'daily conversation' or 'everyday chat'."
        )


# ===================== UI =====================


def create_web_app(api_key=None):
    app = Flask(__name__)
    bot = HybridChatbot(api_key)

    HTML_UI = """
    <!DOCTYPE html><html lang='en'><head>
    <meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/>
    <title>Hybrid AI Chatbot</title>
    <style>
    body{margin:0;padding:0;font-family:sans-serif;background:linear-gradient(135deg,#0ea5e9,#7c3aed);
    display:flex;align-items:center;justify-content:center;min-height:100vh;}
    .container{width:90%;max-width:900px;height:90vh;background:#0b1220;color:#e5e7eb;
    display:flex;flex-direction:column;border-radius:18px;overflow:hidden;box-shadow:0 20px 80px rgba(0,0,0,.4);}
    .chat{flex:1;overflow:auto;padding:20px;background:#0f172a;}
    .bubble{max-width:75%;padding:12px 16px;border-radius:12px;margin:8px 0;white-space:pre-wrap;}
    .user{background:#1e293b;color:#e2e8f0;margin-left:auto;border-bottom-right-radius:4px;}
    .bot{background:#1f2937;border:1px solid #334155;color:#f8fafc;border-bottom-left-radius:4px;}
    .input{display:flex;padding:16px;background:#111827;border-top:1px solid #334155;}
    input{flex:1;padding:12px;border:none;border-radius:10px;background:#0b1220;color:white;outline:none;}
    button{margin-left:10px;background:#2563eb;color:white;border:none;border-radius:10px;padding:12px 18px;cursor:pointer;}
    button:hover{filter:brightness(1.1);}
    </style></head><body>
    <div class='container'><div id='chat' class='chat'>
    <div class='bubble bot'>🤖 Welcome! Type <b>help</b> to see commands.</div></div>
    <div class='input'><input id='msg' placeholder='Type your message...'><button id='send'>Send</button></div></div>
    <script>
    const c=document.getElementById('chat'),i=document.getElementById('msg'),b=document.getElementById('send');
    function add(role,text){const d=document.createElement('div');d.className='bubble '+role;d.textContent=text;c.appendChild(d);c.scrollTop=c.scrollHeight;}
    async function send(){const t=i.value.trim();if(!t)return;add('user',t);i.value='';b.disabled=true;
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});
    const d=await r.json();add('bot',d.response);b.disabled=false;i.focus();}
    b.onclick=send;i.onkeypress=e=>{if(e.key==='Enter'){send();}};
    </script></body></html>
    """

    @app.route("/")
    def index():
        return HTML_UI

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.get_json() or {}
        text = data.get("message", "")
        return jsonify(bot.chat(text))

    return app


app = create_web_app(os.getenv("PERPLEXITY_API_KEY"))

if __name__ == "__main__":
    print("Running Hybrid Chatbot → http://localhost:5000")
    if not os.getenv("PERPLEXITY_API_KEY"):
        print("PERPLEXITY_API_KEY not set (daily conversation disabled).")
    app.run(debug=True, host="0.0.0.0", port=5000)

