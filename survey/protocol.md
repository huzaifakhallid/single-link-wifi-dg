# Survey protocol: single-link and few-antenna cross-domain WiFi gesture and activity recognition

## Scope
Learning-based WiFi recognition of human gestures or activities whose stated aim is to work in a domain
(user, location, orientation, environment, device) that differs from the training domain.
The survey organises these methods by the **resources they need at deployment and training time**:
number of links and antenna pairs, access to target-domain data, and external knowledge (pretraining, synthetic data, large models).

## Sources and searches
See `search/search_log.json` (exact strings, dates, hit counts) and `search/prisma_identification.json`.
OpenAlex replaces Scopus and Web of Science, which were not available to the authors; IEEE Xplore was exported
manually; ACM Digital Library was searched for identification, and ACM records entered through OpenAlex.

## Inclusion criteria (all must hold)
- I1  Published 2021-2026 (journal, conference, or a preprint later accepted at a peer-reviewed venue).
- I2  Uses WiFi signals (CSI, CSI ratio, BFI, RSSI) as the sensing input.
- I3  Recognises human gestures or activities (classification), including sign language and fine-grained gestures.
- I4  Evaluates on at least one domain not seen in training, or proposes a method explicitly for that purpose.
- I5  Full text available in English.

## Exclusion criteria (any one excludes)
- E1  Non-WiFi sensing only (radar, mmWave, UWB, acoustic, vision, wearables), or WiFi only as one of several fused modalities without a WiFi-only result.
- E2  Task is not gesture/activity recognition (localisation, vital signs, crowd counting, authentication only).
- E3  Only in-domain evaluation and no cross-domain claim.
- E4  Not a primary study or review (editorials, abstracts, posters under 4 pages, theses).
- E5  Duplicate or earlier version of an included study (keep the most complete version).

Review papers that meet I1-I3 go to the separate review-comparison set (6-10 reviews), not the primary set.

## Screening procedure
1. Title and abstract: two authors screen `search/screening.csv` independently (R1 and R2 columns:
   include / exclude / unsure, with the criterion code as reason). Disagreements are resolved by discussion
   and recorded in `final_decision`. Agreement is reported as Cohen's kappa.
2. Full text: the same procedure on the papers kept in step 1.
3. Snowballing: backward (references) and forward (citations) from the included studies, one round,
   using the same criteria. Known omissions from the ACM full-text search are expected to surface here.

## Data extraction (per included primary study)
Links and antenna pairs used in training and testing; input representation; model family; domain shift type
evaluated; target-data requirement (none / unlabelled / few labelled); external knowledge used; dataset;
protocol (training domains, held-out domains, model-selection rule); reported metrics and results; code availability.

## Quality criteria (0 / 0.5 / 1 each)
- Q1  The cross-domain protocol is fully specified (which domains train, validate, test).
- Q2  Model selection does not use target-domain labels.
- Q3  Results are averaged over repeated runs or report variance.
- Q4  Compared with at least one baseline under the same protocol.
- Q5  Code or data is publicly available.
- Q6  Reports per-class or per-domain results, not only a single average.
