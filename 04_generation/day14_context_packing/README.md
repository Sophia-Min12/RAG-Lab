# Day 14 · Context Packing: Order, Budget, and the Lost-in-the-Middle Effect

> The first generator in this repo. Everything before today produced a ranked list; this produces a sentence, and a sentence has to be judged rather than looked up.

`google/flan-t5-base`, 248M parameters, 512-token input limit — small enough to run on a laptop CPU, instruction-tuned enough to follow "use only the context".

## The bottleneck has moved

[Day 10](../../02_retrieval/day10_reranking/) established that dense retrieval puts the answer in the top ten for **every** question in this set. So every number below 100% from here on is the generator failing to use something it was handed.

## More context is worse

```
  k   correct   abstained  mean tokens    max  truncated
  1       44%         21%           55     63         no
  2       65%         12%           71     80         no
  3       65%         12%           86     95         no
  5       59%          9%          117    130         no
  8       59%          6%          166    180         no
 12       62%          6%          229    243         no
```

Accuracy peaks at two or three chunks and **falls back** by five — while retrieval recall is rising the whole way. More context is not more information to this model; it is more to be distracted by.

And nothing truncates. The largest prompt here is 243 tokens against a 512-token limit, so this is not the context window being exceeded. It is the model doing worse with material it was successfully given, which is a harder problem than a budget.

### And retrieval is not necessary at all here

```
the ENTIRE handbook is 437 tokens, inside the 512-token limit.

no retrieval at all, whole corpus as context:   56% correct, 18% abstained
dense top-2                                     65% correct, 12% abstained
dense top-3                                     65% correct, 12% abstained
dense top-8                                     59% correct,  6% abstained
```

The whole corpus fits. So retrieval is not *necessary* on this corpus — and it is still worth about nine points, every one of which comes from leaving things **out**. The answer was always available; what retrieval bought was the absence of the other twenty-four chunks.

Worth holding onto for the rest of Level 4: at this size the retriever is a distractor filter rather than a finder, so any claim here about retrieval helping generation is a claim about filtering.

## The lost-in-the-middle effect, measured

The same ten chunks every time. The only thing that changes is where the chunk holding the answer sits:

```
 position   correct   abstained
        0       68%          9%
        2       59%         15%
        4       59%         12%
        6       56%         18%
        9       59%         12%
```

First position is worth about nine points over anywhere else — which is the effect the name refers to, and **only half of it**.

The published shape is a U: strong at the start, weak in the middle, recovering at the end. **There is no recovery here.** The last position scores the same as the middle, so what this model shows is a first-position advantage rather than a U-shape.

That is a different setting rather than a contradiction. Those results are for decoder-only models with contexts of thousands of tokens; this is a 248M encoder-decoder whose encoder attends over 512 tokens bidirectionally. Reporting the U-shape here because it is the famous result would be quoting somebody else's measurement.

And 34 questions means each percentage point is a third of a question. The nine-point gap is three questions, and [Day 11](../../03_evaluation/day11_metrics/)'s interval covers it — so the honest claim is *"first position looks better, on a sample too small to be sure"*.

## Order changes whether the model gives up

```
at k=8:
order          correct   abstained
relevance          59%          6%
reversed           59%         24%
edges              56%         12%
```

The `correct` column barely moves. The `abstained` column moves by a factor of four.

Putting the best chunk **last** leaves accuracy alone and multiplies "not stated" several times over. Order does not change what this model can find so much as how readily it gives up looking.

A system reporting only accuracy would call these three arrangements equivalent. They are not, and the difference is the one a user would notice first.

`edges` — best first, second-best last, the rest buried — is what the lost-in-the-middle papers recommend. It is not better than simply handing over the ranking, and the position table explains why: there is no end-of-context advantage here to exploit.

## Run it

```bash
pip install transformers        # alongside sentence-transformers

python packing.py         # demo (slow: it generates ~300 answers)
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

Packing and answer-scoring are tested without any model. Only the position experiment needs the generator, and it runs on a subset.

> ⚠ `answer_is_correct` uses containment, so it cannot catch an answer that is correct *and* says three wrong things. A test pins that limitation — Day 16 is where it starts to matter.

## Where this leads

Ten days of retrieval work put the answer in front of the model, and it answers about two thirds. Every remaining failure is a generation failure, and none of it shows up in a retrieval metric. **Day 15** makes the answer point at its evidence.
