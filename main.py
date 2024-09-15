from production import forward_chain, populate, IF, AND, THEN, RuleExpression, is_hypothesis
from rules_example_zookeeper import ZOOKEEPER_RULES, ZOOKEEPER_DATA
import re

if __name__ == '__main__':
    rules = ZOOKEEPER_RULES
    data = ZOOKEEPER_DATA

    while True:
        print('Would you like forward of backward chaining?')
        res = input().lower()

        if res == 'f' or res == 'forward' or res == 'forward chaining':
            chain, facts = forward_chain(
                rules=rules,
                data=data,
                apply_only_one=False,
                verbose=False
            )

            hypothesis = None

            for f in chain:
                if is_hypothesis(rules, re.sub("^X", "(?x)", f)):
                    hypothesis = f
                    break

            inter_facts = list(set(chain) - set(facts) - { hypothesis })

            print(facts)
            print(inter_facts)
            print(hypothesis)

            break

        elif res == 'b' or res == 'backward' or res == 'backward chaining':
            break
