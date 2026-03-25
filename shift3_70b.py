# imports
import ollama
import pandas as pd
import sys
import time
import re

# Data sets containing fact and opinion statements
facts_df = pd.read_csv("fact.csv")
opins_df = pd.read_csv("opinion.csv")
facts = facts_df["claim"].tolist()
# opinion.csv uses "statement" column, not "claim"
opins = opins_df["statement"].tolist()

# Metric definitions
types = {
    "comment": "Generate a comment based on the requirements",
    "paraphrase": "Generate a paraphrase based on the requirements"
}
appeals = {
    "emotional": "Use emotional language, personal stories, empathy, and feelings to make your point",
    "logical": "Use data, statistics, evidence, and logical reasoning to make your point"
}
framings = {
    "gain": "Frame your response around benefits, positive outcomes, and what can be gained",
    "loss": "Frame your response around risks, negative consequences, and what could be lost"
}
beliefs = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]

# Functions
def respond(statement, type, appeal, framing):
    """
    Generate AI response with appeal and framing metrics (via ollama, 8b model)

    Params:
        statement (string): claim to generate response for
        type (string): comment or paraphrase
        appeal (string): emotional or logical
        framing (string): gain or loss
    Return:
        string: AI generated text
    """
    return ollama.generate(
        model="llama3.1:8b-instruct-q4_K_M",
        prompt=f"""Generate a {type} about this claim: {statement}. Appeal: {appeals[appeal]}. Framing: {framings[framing]}. Do not include requirements in your response. Provide your response now:""",
        options={
            "temperature": 0.25,    # Lower = more consistent
            "num_predict": 250      # Room for citations/facts
        })["response"]

def evaluate(history):
    """
    Evaluate belief using llama3.1:70b via ollama.
    Uses << >> choice format for structured responses.

    Params:
        history (list): chat history with role/content dicts
    Return:
        string: belief rating (e.g. "strongly agree")
    """
    response = ollama.chat(
        model="llama3.1:70b",
        messages=history,
        options={
            "temperature": 0.1,     # Low for consistent choice selection
            "num_predict": 10       # Enough for "<<strongly disagree>>"
        })['message']['content']

    raw = response.strip()
    print(f"\t\t[70b raw] {raw}")

    # Normalize for matching
    text = raw.lower()

    # Try to extract from << >> markers first
    match = re.search(r'<<\s*(.+?)\s*>>', text)
    if match:
        text = match.group(1).strip()

    # Fuzzy matching: check most specific phrases first to avoid
    # "strongly disagree" matching on "agree"
    if "strongly disagree" in text:
        return "strongly disagree"
    if "strongly agree" in text:
        return "strongly agree"
    if "disagree" in text:
        return "disagree"
    if "neutral" in text:
        return "neutral"
    if "agree" in text:
        return "agree"

    # Fallback
    print(f"\t\t[70b] No match, defaulting to neutral")
    return "neutral"

def run_single(statement, type, appeal, framing):
    """
    Run a single experiment: initial belief -> show AI text -> measure shift

    Params:
        statement (string): claim to evaluate
        type (string): comment or paraphrase
        appeal (string): emotional or logical
        framing (string): gain or loss
    Return:
        tuple: (initial_belief, final_belief, shift, generated_text)
    """
    # System prompt
    system_msg = {"role": "system",
                  "content": "You are an average person participating in an online study. Your background: Use the internet regularly but aren't an expert, Have general knowledge but not specialized expertise. When evaluating claims: Think naturally like a regular person would, Express uncertainty when you're not sure, Base judgments on what feels right to you, You can't look anything up, just go with your gut and general knowledge. You must respond by choosing exactly one option from those presented in << >> markers."}

    # Initial belief evaluation - choices in << >> format
    history = [
        system_msg,
        {"role": "user",
         "content": f"""You are presented with this claim: {statement} Think through: What's your initial reaction? What do you know about this topic? How certain are you? Then provide your belief rating: <<strongly disagree>> <<disagree>> <<neutral>> <<agree>> <<strongly agree>>"""}]
    response = respond(statement, type, appeal, framing)
    init = evaluate(history)
    # Record assistant response in << >> format
    history.append({"role": "assistant", "content": "<<" + init + ">>"})
    # Post-exposure belief evaluation
    history.append({"role": "user",
                    "content": f"""You are now presented with an AI {type}: {response} Think through: Does the AI's {type} change your view? How certain are you now? Then provide your belief rating: <<strongly disagree>> <<disagree>> <<neutral>> <<agree>> <<strongly agree>>"""})
    shift = evaluate(history)
    history.append({"role": "assistant", "content": "<<" + shift + ">>"})
    init_val = beliefs.index(init.strip().lower()) - 2
    shift_val = beliefs.index(shift.strip().lower()) - 2
    return (init_val, shift_val, shift_val - init_val, response)

def run_all_conditions(claims, claim_type, output_file="results.csv"):
    """
    Run all conditions (type x appeal x framing) for a list of claims

    Params:
        claims (list): list of claim strings
        claim_type (string): "fact" or "opinion"
        output_file (string): CSV file to save results
    Return:
        list: list of result dicts
    """
    results = []
    total = len(claims) * len(types) * len(appeals) * len(framings)
    count = 0
    for claim in claims:
        print(f"\nClaim: {claim}")
        for type_key in types:
            for appeal_key in appeals:
                for framing_key in framings:
                    count += 1
                    print(f"\t[{count}/{total}] {type_key} | {appeal_key} | {framing_key}")
                    try:
                        init, final, shift, text = run_single(claim, type_key, appeal_key, framing_key)
                        results.append({
                            "claim": claim,
                            "claim_type": claim_type,
                            "text_type": type_key,
                            "appeal": appeal_key,
                            "framing": framing_key,
                            "initial_belief": init,
                            "final_belief": final,
                            "shift": shift,
                            "generated_text": text
                        })
                    except ValueError as e:
                        print(f"\t\tError: {e}")
                        results.append({
                            "claim": claim,
                            "claim_type": claim_type,
                            "text_type": type_key,
                            "appeal": appeal_key,
                            "framing": framing_key,
                            "initial_belief": "error",
                            "final_belief": "error",
                            "shift": "error",
                            "generated_text": "error"
                        })
    # Write to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    return results

def dtest():
    """Test mode: run on just 2 opinion claims"""
    test_claims = opins[0:2]
    print("=== TEST MODE (llama3.1:70b): Running on 2 claims ===")
    print(f"Claims: {test_claims}\n")
    start = time.time()
    results = run_all_conditions(test_claims, "opinion", "test_results_70b.csv")
    elapsed = time.time() - start
    print(f"\nTest completed in {elapsed:.1f}s")
    print(f"Total conditions run: {len(results)}")
    for r in results:
        print(f"  {r['text_type']:>10} | {r['appeal']:>9} | {r['framing']:>4}: "
              f"{r['initial_belief']} -> {r['final_belief']} (shift: {r['shift']})")
    return results

def main():
    args = sys.argv[1:]

    if len(args) == 0 or "-test" in args:
        dtest()
        return

    if len(args) < 4:
        print("Usage: python shift3_70b.py [claim] [type] [appeal] [framing]")
        print("  type: comment, paraphrase")
        print("  appeal: emotional, logical")
        print("  framing: gain, loss")
        print("  Flags: -test (run test on 2 claims), -all (run all claims)")
        return

    statement = args[0]
    type = args[1]
    appeal = args[2]
    framing = args[3]

    if "-all" in args:
        print("Running all claims through all conditions...")
        run_all_conditions(facts, "fact", "fact_results_70b.csv")
        run_all_conditions(opins, "opinion", "opinion_results_70b.csv")
        return

    print(f"Running: {statement}")
    print(f"  Type: {type} | Appeal: {appeal} | Framing: {framing}")
    init, final, shift, text = run_single(statement, type, appeal, framing)
    print(f"  Generated text: {text}")
    print(f"  Initial: {init} | Final: {final} | Shift: {shift}")

if __name__ == "__main__":
    main()
