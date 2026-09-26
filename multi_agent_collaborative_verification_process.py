ARCHITECTURES = (
    "star", "chain", "tree", "debate", "hypothesis_verify", "storyline",
)
ROLES = ("global timeline", "local segment 1", "local segment 2", "local segment 3")


def packets(reports):
    return ["Observer %d (%s):\n%s" % (i, ROLES[i], text)
            for i, text in enumerate(reports)]


def execute_graph(name, reports, generate, choose):
    if name not in ARCHITECTURES or len(reports) != 4:
        raise ValueError("Expected a supported architecture and four reports")
    p = packets(reports)

    if name == "star":
        summary = generate("star_reasoner",
            "Integrate these four observer reports into one question-relevant evidence "
            "report. Preserve source IDs, timestamps, temporal relations, conflicts and "
            "uncertainty. Distinguish observations from inferences; do not invent facts.", p)
        final = [summary]
    elif name == "chain":
        carry = p[0]
        for i in range(1, 4):
            carry = generate("chain_%d" % i,
                "Merge the prior evidence with your own report. Preserve source IDs, "
                "times, contradictions, and facts needed to distinguish the options. "
                "Do not invent observations.", [carry, p[i]])
        final = [carry]
    elif name == "tree":
        final = [generate("pair_%d" % i,
            "Combine these two reports, preserving decisive facts, source IDs, "
            "timestamps and uncertainty. Do not resolve conflicts by invention.", p[i:i+2])
            for i in (0, 2)]
    elif name == "debate":
        final = [generate("debater_%d" % i,
            "You represent observer %d. Review the other reports, challenge unsupported "
            "claims and revise your option assessment using cited evidence. Distinguish "
            "your own observation from peer claims. This is one synchronous debate round." % i,
            [p[i]] + [p[j] for j in range(4) if j != i]) for i in range(4)]
    elif name == "hypothesis_verify":
        hypothesis = generate("proposer",
            "Propose the most plausible answer and an alternative. Identify concrete "
            "reported evidence that would distinguish them.", p)
        critique = generate("verifier",
            "Check the proposed hypotheses against each source report. Identify "
            "contradictions, missing evidence and unsupported inferences. Return a "
            "verification judgement with source references.", p + [hypothesis])
        final = p + [hypothesis, critique]
    elif name == "storyline":
        timeline = generate("timeline",
            "Construct a concise chronological outline relevant to the question, "
            "preserving observed times and stating gaps.", [p[0]])
        final = [timeline] + [generate("story_local_%d" % i,
            "Check this local report against the global outline. Resolve temporal "
            "relations using only reported evidence; preserve conflicting observations "
            "and distinguish them from inferred storyline.", [p[i], timeline])
            for i in (1, 2, 3)]
    return choose(final, routing=False)


def run_all(reports, generate, choose):
    return {
        name: execute_graph(name, reports, generate, choose)
        for name in ARCHITECTURES
    }


def score(gold, direct, predicted, observers):
    n = len(gold)
    if not n or any(len(x) != n for x in (direct, predicted, observers)):
        raise ValueError("Expected nonempty, complete paired answers")
    if any(len(row) != 4 for row in observers):
        raise ValueError("Expected four observer answers per question")
    hit = [p == y for p, y in zip(predicted, gold)]
    base = [d == y for d, y in zip(direct, gold)]
    oracle = [y in row for y, row in zip(gold, observers)]
    return {
        "n": n,
        "accuracy_percent": 100 * sum(hit) / n,
        "delta_direct_pp": 100 * (sum(hit) - sum(base)) / n,
        "corrected": sum(h and not b for h, b in zip(hit, base)),
        "harmed": sum(b and not h for h, b in zip(hit, base)),
        "observer_correct_final_wrong": sum(o and not h for o, h in zip(oracle, hit)),
        "all_observers_wrong_final_correct": sum(h and not o for o, h in zip(oracle, hit)),
    }


if __name__ == "__main__":
    print("Methods:", ", ".join(ARCHITECTURES))
