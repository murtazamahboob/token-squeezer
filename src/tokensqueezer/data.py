"""Bundled benchmark samples: verbose real-world prompts with ground truth.

Each sample has:

* ``text`` — the verbose original prompt;
* ``keywords`` — essential terms that must survive compression;
* ``facts`` — short key-fact statements whose content must survive.

The samples are deliberately wordy, repetitive and filler-heavy, the way
real prompts written in a hurry look. Nothing here is fetched or generated
at runtime; the benchmark is deterministic.
"""

from __future__ import annotations

SAMPLES: list[dict] = [
    {
        "id": "product-description",
        "title": "Verbose product description",
        "text": (
            "We are building a really amazing artificial intelligence writing assistant that is "
            "actually designed to help users write better content in a very fast and efficient way. "
            "The artificial intelligence writing assistant uses machine learning and deep learning "
            "to understand what the user wants to say. It is really important to note that the "
            "artificial intelligence writing assistant is very easy to use and it is basically "
            "suitable for everyone. In order to get started, the user just needs to sign up for an "
            "account and then they can start writing right away. The product is available on a "
            "daily basis and it works on all devices. Due to the fact that many users are very "
            "busy, the artificial intelligence writing assistant saves a large number of hours "
            "every week. The artificial intelligence writing assistant is the best writing assistant."
        ),
        "keywords": ["AI", "writing", "assistant", "machine learning", "sign up", "devices", "hours"],
        "facts": [
            "AI writing assistant helps users write better content",
            "uses machine learning and deep learning",
            "user signs up for an account to start",
            "works on all devices",
            "saves hours every week",
        ],
    },
    {
        "id": "meeting-notes",
        "title": "Rambling meeting notes to action items",
        "text": (
            "So in today's meeting we basically talked about the launch plan and we really need to "
            "get the launch plan finished. Sarah said she would actually handle the marketing copy "
            "and she said she would have it ready by Friday. Sarah will handle the marketing copy "
            "by Friday. John mentioned that the pricing page is still not done and John said the "
            "pricing page needs to be done before the launch. The pricing page must be done before "
            "the launch. We also talked about the beta feedback and the beta feedback was actually "
            "very positive overall. In order to move forward, we need to fix the three bugs that "
            "were reported in the beta feedback. The three bugs from the beta feedback need to be "
            "fixed. The next meeting is on Monday at 10am and the next meeting will be on Monday."
        ),
        "keywords": ["launch", "Sarah", "marketing", "Friday", "John", "pricing", "beta", "bugs", "Monday"],
        "facts": [
            "Sarah handles marketing copy by Friday",
            "John finishes the pricing page before launch",
            "beta feedback was positive",
            "three bugs from beta feedback must be fixed",
            "next meeting Monday at 10am",
        ],
    },
    {
        "id": "support-system-prompt",
        "title": "Overlong support-bot system prompt",
        "text": (
            "You are a customer support assistant. You are a customer support assistant that helps "
            "customers with their questions. You should always be polite and you should always be "
            "helpful. It is very important that you never share internal information with customers. "
            "You must never share internal information. If a customer asks about a refund, you should "
            "explain the refund policy. The refund policy allows refunds within 30 days of purchase. "
            "Refunds are allowed within 30 days of purchase. If a customer is angry, you should stay "
            "calm and you should apologize for the inconvenience. Please apologize when customers are "
            "upset. You should use simple language and you should avoid technical jargon. Always use "
            "simple language. If you do not know the answer, you should say that you do not know and "
            "you should offer to escalate the issue to a human agent. Escalate to a human agent when "
            "you do not know the answer."
        ),
        "keywords": ["support", "polite", "refund", "30 days", "escalate", "human agent", "simple language"],
        "facts": [
            "assistant is a customer support assistant",
            "never share internal information",
            "refunds allowed within 30 days of purchase",
            "apologize when customers are upset",
            "escalate to a human agent when answer unknown",
        ],
    },
    {
        "id": "research-question",
        "title": "Research question with repeated context",
        "text": (
            "I am writing a research paper about climate change and I need help with my research "
            "paper about climate change. The research paper is about how climate change affects "
            "agriculture in South Asia. Climate change affects agriculture in South Asia in many "
            "ways. Specifically, I want to know how rising temperatures and changing rainfall "
            "patterns affect rice yields in Pakistan and India. Rising temperatures and changing "
            "rainfall patterns are the key factors. Please summarize the three most important "
            "recent studies on this topic. I need the three most important recent studies. For each "
            "study, please give the authors, the year, the main finding, and one limitation. The "
            "summary should be in plain language and it should be suitable for a non-expert reader."
        ),
        "keywords": ["climate change", "agriculture", "South Asia", "rice", "Pakistan", "India", "studies", "plain language"],
        "facts": [
            "paper is about climate change and agriculture in South Asia",
            "focus on rice yields in Pakistan and India",
            "rising temperatures and changing rainfall patterns",
            "summarize three most important recent studies",
            "give authors, year, main finding, limitation per study",
        ],
    },
    {
        "id": "code-review-request",
        "title": "Verbose code review request",
        "text": (
            "Hi, I would really like you to review my Python code for me. The code is a Flask web "
            "application and it is basically a simple to-do list application built with Flask. The "
            "application has a bug where deleting a task sometimes deletes the wrong task. Deleting "
            "a task sometimes deletes the wrong task and this is very confusing for users. Please "
            "look at the code very carefully and find the bug. I need you to find the bug that "
            "causes the wrong task to be deleted. Please also check for security issues in the code. "
            "In particular, please check for SQL injection vulnerabilities and please check for "
            "cross-site scripting vulnerabilities. The code uses SQLite as the database and the code "
            "uses raw SQL queries. Please suggest a fix for the bug and please suggest fixes for any "
            "security issues you find. Please explain your suggestions in simple terms."
        ),
        "keywords": ["Python", "Flask", "to-do", "deleting", "wrong task", "SQL injection", "cross-site scripting", "SQLite", "fix"],
        "facts": [
            "Flask to-do list application in Python",
            "bug deletes the wrong task",
            "check for SQL injection vulnerabilities",
            "check for cross-site scripting vulnerabilities",
            "uses SQLite with raw SQL queries",
            "suggest a fix and explain simply",
        ],
    },
    {
        "id": "support-context-dump",
        "title": "Customer context dump with repeated order details",
        "text": (
            "Customer: Ayesha Khan, order number 88412, order number 88412. The customer ordered a "
            "wireless headset on 12 October and the order number is 88412. The headset arrived "
            "damaged and the customer wants a replacement. The customer wants a replacement headset, "
            "not a refund. The customer does not want a refund. The order was placed on 12 October "
            "and the delivery address is Lahore, Pakistan. The customer has contacted support twice "
            "already and the customer is frustrated. The customer contacted support twice before "
            "this conversation. Please draft a reply that apologizes for the damaged headset, "
            "confirms a free replacement will be shipped within 3 business days, and asks the "
            "customer to confirm the delivery address in Lahore. The replacement will be shipped "
            "within 3 business days."
        ),
        "keywords": ["Ayesha Khan", "88412", "headset", "replacement", "12 October", "Lahore", "3 business days"],
        "facts": [
            "customer Ayesha Khan, order 88412",
            "wireless headset arrived damaged",
            "customer wants a replacement, not a refund",
            "delivery address is Lahore, Pakistan",
            "free replacement shipped within 3 business days",
            "customer contacted support twice before",
        ],
    },
]
