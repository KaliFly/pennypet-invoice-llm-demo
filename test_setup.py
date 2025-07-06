# test_setup.py
from openrouter_client import OpenRouterClient

def test_clients():
    for key in ["primary", "secondary"]:
        client = OpenRouterClient(model_key=key)
        resp = client.chat([
            {"role": "system", "content": '{"hello":"world"}'},
            {"role": "user",   "content": "Test"}
        ])
        print(f"[{key}] Réponse :", resp.choices[0].message.content)

if __name__ == "__main__":
    test_clients()
