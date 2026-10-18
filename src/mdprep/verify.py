"""Qualify current local corpus artifacts for experimental training.

Regenerate with ``python -m mdprep.prepare --root PATH --part all``.
Run ``python -m mdprep.verify --root PATH`` after assembly. The latter command
requires local pinned raw sources/font, ffmpeg and ffprobe. Any unresolved
glyph recognition or geometry remains a failed automated gate. Only a fully successful run
atomically publishes data/verification-contract.json. Failed runs write
data/verification-report.json without replacing the previous contract.

Packaged readers call validate_verification_contract without upstream caches.
Project scope and row_receipts to the included primary splits, retain the source
receipt, and recompute qualification_sha256 after projection. Keep the original
contract hash/count/scope in the qualification object. Hashes bind local
artifacts; they are not cryptographic signatures from an external authority.
Human review and independent photo visible truth remain unperformed.
"""

import argparse
import collections
import hashlib
import json
import math
import subprocess
from pathlib import Path

from PIL import Image

from mdprep.assemble import inspect_video, load_rows, media_paths
from mdprep.common import PART_QUOTAS, SPLITS, sha256_file, stable_id, write_json
from mdprep.records import validate_record
from mdprep.rule_variants import recompute_label as recompute_rule_variant_label
from mdprep.shortcut_check import check as generated_shortcut_check
from mdprep.text_catalog import HELD_OUT_INTENTS
from mdprep.verify_sources import answer_key, reconcile, source_tables, text_answer
from mdprep.verify_video import extract_frames, video_answer
from mdprep.verify_visuals import extract_visual, visual_answer


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def confined(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unconfined artifact: {relative}")
    return path


def candidate_only_probe(rows):
    """Repeat the original description-only train-majority/development control."""
    training=collections.defaultdict(collections.Counter)
    prior=collections.Counter()
    development=[]
    train_questions=0
    for row in rows:
        if row["provenance"]["source"]!="banking77" or row["split"] not in {"train","dev"}:
            continue
        criteria=row["question"]["criteria"]
        key=tuple(sorted(criteria.values()))
        target=criteria[row["target"]["label"]]
        if len(key)!=5 or len(set(key))!=5: raise ValueError("Malformed BANKING77 candidates")
        if row["split"]=="train":
            training[key][target]+=1
            prior[target]+=1
            train_questions+=1
        else:
            development.append((key,target))
    if not train_questions or not development: raise ValueError("Candidate-only control requires train and dev")
    correct=0
    known=0
    for key,target in development:
        distribution=training.get(key)
        known+=distribution is not None
        scores=distribution if distribution is not None else prior
        prediction=sorted(key,key=lambda description:(-scores.get(description,0),description))[0]
        correct+=prediction==target
    return {"scope":"BANKING77 development, candidate descriptions only",
            "method":"candidate-set majority from train; global class prior for unseen sets",
            "train_questions":train_questions,"dev_questions":len(development),
            "unique_training_candidate_sets":len(training),"known_sets_in_dev":known,
            "correct":correct,"accuracy":correct/len(development),"chance_accuracy":0.2,
            "state_or_audit_used":False,"final_test_used":False}


def validate_verification_contract(root: Path, path: Path | None = None) -> dict:
    """Validate a hash-bound qualification receipt without needing upstream raw labels."""
    root=Path(root).resolve()
    contract=json.loads((path or root/"data/verification-contract.json").read_text())
    if contract.get("status")!="VERIFIED" or contract.get("all_questions_checked") is not True:
        raise ValueError("Data qualification is not successful")
    scope=contract["scope"]
    if not scope or len(scope)!=len(set(scope)) or not set(scope)<=set(SPLITS):
        raise ValueError("Invalid qualification scope")
    if set(contract["prepared_hashes"])!=set(scope): raise ValueError("Qualification split hashes do not match scope")
    if not contract.get("source_reconciliation") or object_hash(contract["source_reconciliation"])!=contract.get("source_receipt_sha256"):
        raise ValueError("Source reconciliation receipt is missing or changed")
    if contract["source_reconciliation"].get("status")!="VERIFIED": raise ValueError("Raw source reconciliation was not verified")
    if contract.get("human_review",{}).get("status")!="NOT_VERIFIED": raise ValueError("Software cannot certify human review")
    unsigned={key:value for key,value in contract.items() if key!="qualification_sha256"}
    if contract.get("qualification_sha256")!=object_hash(unsigned):
        raise ValueError("Qualification receipt hash mismatch")
    receipts=contract["row_receipts"]
    if not isinstance(receipts,dict): raise ValueError("Qualification row receipts must be a mapping")
    identifiers=set()
    train_identifiers=set()
    media=set()
    for split,digest in contract["prepared_hashes"].items():
        prepared=confined(root/"data",f"prepared/{split}.jsonl")
        if sha256_file(prepared)!=digest: raise ValueError(f"Prepared split changed: {split}")
        for row in load_rows(prepared):
            validate_record(row)
            if row["split"]!=split or row["id"] in identifiers: raise ValueError("Invalid prepared identity")
            identifiers.add(row["id"])
            if split=="train": train_identifiers.add(row["id"])
            if receipts.get(row["id"])!=object_hash(row): raise ValueError("Qualified row changed")
            media.update(media_paths(row))
    if len(identifiers)!=contract["questions_checked"]: raise ValueError("Qualification question count changed")
    if set(receipts)!=identifiers: raise ValueError("Qualification row receipt scope changed")
    if set(contract["media_hashes"])!=media: raise ValueError("Qualification media scope changed")
    for relative,digest in contract["media_hashes"].items():
        if sha256_file(confined(root/"data",relative))!=digest: raise ValueError(f"Qualified media changed: {relative}")
    derived_sizes={"prepared/pilot.jsonl":1000,"prepared/diagnostic.jsonl":64}
    for relative,digest in contract.get("derived_hashes",{}).items():
        if relative not in derived_sizes or not train_identifiers:
            raise ValueError("Unknown or non-training derived subset")
        if sha256_file(confined(root/"data",relative))!=digest: raise ValueError(f"Derived artifact changed: {relative}")
        subset=load_rows(root/"data"/relative)
        subset_ids=[row["id"] for row in subset]
        if len(subset)!=min(derived_sizes[relative],len(train_identifiers)) or len(set(subset_ids))!=len(subset_ids):
            raise ValueError("Derived training subset has the wrong size or duplicate IDs")
        for row in subset:
            if row["split"]!="train" or row["id"] not in train_identifiers or receipts.get(row["id"])!=object_hash(row):
                raise ValueError("Derived subset contains a changed or unqualified training row")
    for relative,digest in contract.get("verifier_hashes",{}).items():
        if sha256_file(confined(root,relative))!=digest: raise ValueError(f"Qualification implementation changed: {relative}")
    if isinstance(contract.get("qualification"),dict):
        original=contract["qualification"]
        if original.get("questions_checked",0)<len(identifiers) or not set(scope)<=set(original["scope"]) or not original.get("contract_sha256"):
            raise ValueError("Invalid projected qualification provenance")
    return contract


def _isolate(mapping,key,identity,name):
    previous=mapping.setdefault(key,identity)
    if previous!=identity: raise ValueError(f"{name} assigned to different source groups")


def reconcile_temporal_frames(row, facts):
    declared=row["audit"].get("frame_facts")
    if len(facts)!=4 or not isinstance(declared,list) or len(declared)!=4 or any(not isinstance(frame,dict) for frame in declared):
        raise ValueError("Temporal reconciliation requires exactly four declared and decoded frames")
    expected_times=[index/row["videos"][0]["fps"] for index in range(4)]
    times=row["audit"].get("sample_times_s")
    if not isinstance(times,list) or len(times)!=4 or any(type(t) not in (int,float) or not math.isfinite(t) or not math.isclose(t,expected,abs_tol=1e-6,rel_tol=0) for t,expected in zip(times,expected_times,strict=True)):
        raise ValueError("Temporal sample times do not match the decoded profile")
    canonical=[tuple((key,tuple(value) if isinstance(value,(list,tuple)) else value) for key,value in sorted(fact.items())) for fact in facts]
    for index,(fact,frame) in enumerate(zip(facts,declared,strict=True)):
        time_value=frame.get("time_s")
        if type(frame.get("sample_index")) is not int or frame["sample_index"]!=index or type(time_value) not in (int,float) or not math.isfinite(time_value) or not math.isclose(time_value,expected_times[index],abs_tol=1e-6,rel_tol=0):
            raise ValueError("Temporal declared frame order or time changed")
        for key,value in fact.items():
            if key.endswith("color"):
                if value!=frame.get(key): raise ValueError("Decoded color mismatch")
            elif tuple(value)!=tuple(frame.get(key,())):
                raise ValueError(f"Decoded center mismatch: {key} observed={value} declared={frame.get(key)}")
    return canonical


def _source_assets(root):
    sources={}
    indexes={}
    for name in ("banking77", "open_images_v7_photos", "visuals", "videos", "hard-v1", "rule-variants-v1"):
        path=root/"data/sources"/f"{name}.json"
        document=json.loads(path.read_text())
        sources[path.relative_to(root).as_posix()]=sha256_file(path)
        assets=document.get("assets",document.get("attributions",[]))
        indexes[name]={a.get("path",a.get("processed_path")):a for a in assets}
    return sources,indexes


def verify(root: Path) -> dict:
    root=Path(root).resolve()
    result={"status":"NOT_VERIFIED","all_questions_checked":False,"checks":{},"errors":[],
            "human_review":{"status":"NOT_VERIFIED","performed":False,"required_for":"production or independently adjudicated visible photo truth"},
            "photo_visible_truth":{"status":"NOT_VERIFIED","note":"Raw annotation fidelity is not independent visible-scene truth."}}
    counts=collections.Counter()
    bank_sets=collections.defaultdict(collections.Counter)
    identities={name:{} for name in ("group","media","evidence","text","creator","original","landing","orbit")}
    media_hashes={}
    rows=[]
    failed_assets=set()
    visual_cache={}
    visual_evidence={}
    frame_cache={}
    decoded=set()
    hardv1_splits = collections.Counter()
    rule_variant_splits = collections.Counter()
    receipts={}
    try:
        source_hashes,indexes=_source_assets(root)
        tables=source_tables(root)
        raw_hashes={p.relative_to(root).as_posix():sha256_file(p) for base in (root/"data/raw/banking77",root/"data/raw/open_images_v7") for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".csv",".json",".html",".txt"} and "images" not in p.parts}
        hardv1_verifier_path=root/".audit/hardv1-independent-verification.json"
        hardv1_verification=json.loads(hardv1_verifier_path.read_text())
        if hardv1_verification.get("status")!="PASSED" or hardv1_verification.get("faults")!=0:
            raise ValueError("Independent hard-v1 source conversion verification did not pass")
        if hardv1_verification.get("verifier_sha256")!=sha256_file(root/"tools/verify_hardv1_conversion.py"):
            raise ValueError("Independent hard-v1 verifier changed after its receipt")
        hardv1_conversion_path=root/".audit/hardv1-conversion.json"
        if hardv1_verification.get("conversion_receipt_sha256")!=sha256_file(hardv1_conversion_path):
            raise ValueError("hard-v1 conversion receipt changed after independent verification")
        for split in ("train","dev"):
            converted=root/".audit/hardv1-converted"/f"{split}.jsonl"
            if hardv1_verification["splits"][split]["converted_sha256"]!=sha256_file(converted):
                raise ValueError(f"Independently verified hard-v1 {split} conversion changed")
        hardv1_admission_path=root/".audit/hardv1-admission.json"
        hardv1_admission=json.loads(hardv1_admission_path.read_text())
        if (hardv1_admission.get("status")!="VERIFIED"
                or hardv1_admission.get("part_sha256")!=sha256_file(root/"data/parts/hardv1.jsonl")
                or hardv1_admission.get("source_manifest_sha256")!=sha256_file(root/"data/sources/hard-v1.json")
                or hardv1_admission.get("splits")!={"train":7876,"dev":464,"calibration":464}
                or hardv1_admission.get("upstream_test_admitted") is not False):
            raise ValueError("hard-v1 admission receipt is stale or incomplete")
        rule_variant_receipt_paths=(
            root/".audit/rule-variants-v1/generation.json",
            root/".audit/rule-variants-v1/leakage-check.json",
            root/".audit/rule-variants-v1/shortcut-check.json",
            root/".audit/rule-structures-heldout/manifest.json",
        )
        generation,leakage,shortcut,heldout=(
            json.loads(path.read_text()) for path in rule_variant_receipt_paths
        )
        wording=generation.get("wording_lint",{})
        if (generation.get("status")!="VERIFIED"
                or generation.get("part_sha256")!=sha256_file(root/"data/parts/rule_variants.jsonl")
                or generation.get("source_manifest_sha256")!=sha256_file(root/"data/sources/rule-variants-v1.json")
                or generation.get("splits")!={"train":5400,"dev":350,"calibration":350}
                or wording.get("status")!="VERIFIED"
                or any(wording.get("forbidden_phrase_counts",{}).values())
                or wording.get("numeric_comparisons_without_inclusivity")!=0
                or wording.get("numeric_boundary_coverage_pass") is not True
                or leakage.get("status")!="VERIFIED" or leakage.get("overlap_count")!=0
                or shortcut.get("status")!="VERIFIED"
                or heldout.get("status")!="VERIFIED" or heldout.get("pairs")!=300
                or heldout.get("compilation",{}).get("status")!="VERIFIED"):
            raise ValueError("rule-variants admission receipts are stale or incomplete")
        part_rows=[]
        for part in ("text","photos","visuals","videos","hardv1","rule_variants"):
            selected=load_rows(root/"data/parts"/f"{part}.jsonl")
            actual=collections.Counter((r["split"],r["modality"]) for r in selected)
            mappings=[("visual_image","image"),("visual_mixed","mixed")] if part=="visuals" else [(part,{"text":"text","photos":"image","videos":"video","hardv1":"text","rule_variants":"text"}[part])]
            expected=collections.Counter({(split,modality):count for quota,modality in mappings for split,count in PART_QUOTAS[quota].items()})
            if actual!=expected: raise ValueError(f"Part quota mismatch: {part}")
            part_rows.extend(selected)
        part_identity={r["id"]:object_hash(r) for r in part_rows}
        if len(part_identity)!=len(part_rows): raise ValueError("Duplicate IDs in source parts")
        prepared_hashes={}
        for split in SPLITS:
            path=root/"data/prepared"/f"{split}.jsonl"
            prepared_hashes[split]=sha256_file(path)
            selected=load_rows(path)
            if any(r["split"]!=split for r in selected): raise ValueError("Prepared split field mismatch")
            rows.extend(selected)
        if {r["id"]:object_hash(r) for r in rows}!=part_identity or len(rows)!=len(part_rows): raise ValueError("Prepared data differs from source parts")
        ids=set()
        for row in rows:
            try:
                validate_record(row)
                if row["id"] in ids: raise ValueError("Duplicate record ID")
                ids.add(row["id"])
                provenance=row["provenance"]
                source=provenance["source"]
                identity=(row["split"],row["group_id"])
                _isolate(identities["group"],row["group_id"],identity,"Source group")
                if provenance.get("upstream_split") in {"test","validation","dev"} and row["split"] in {"train","dev","calibration","policy"}: raise ValueError("Upstream evaluation data leaked")
                signatures=[]
                for relative in media_paths(row):
                    path=confined(root/"data",relative)
                    if relative not in media_hashes: media_hashes[relative]=sha256_file(path)
                    digest=media_hashes[relative]
                    _isolate(identities["media"],digest,identity,"Actual media bytes")
                    signatures.append(digest)
                    if relative not in decoded:
                        if row["videos"]:
                            actual=inspect_video(path)
                            declared=next(v for v in row["videos"] if v["path"]==relative)
                            if any(not math.isclose(actual[k],declared[k],abs_tol=0.0001,rel_tol=0.0001) for k in ("fps","duration_s","frame_count")): raise ValueError("Video metadata mismatch")
                        else:
                            with Image.open(path) as image: image.load()
                        decoded.add(relative)
                normalized=" ".join(row["state"].casefold().split())
                _isolate(identities["evidence"],stable_id(normalized,signatures),identity,"Canonical evidence")
                if row["modality"]=="text": _isolate(identities["text"],normalized,identity,"Normalized text")
                target=answer_key(row)
                if source=="banking77":
                    reconcile(row,tables)
                    key=tuple(sorted(row["audit"]["original_candidate_keys"]))
                    if len(key)!=5: raise ValueError("BANKING77 must have five intents per candidate set")
                    if any((label in HELD_OUT_INTENTS)!=(row["split"]=="test_ood") for label in key): raise ValueError("BANKING77 candidate holdout leakage")
                    bank_sets[(row["split"],key)][target]+=1
                    counts["raw_label_fidelity"]+=1
                elif source=="open_images_v7_train_human_verified":
                    reconcile(row,tables)
                    asset=indexes["open_images_v7_photos"][row["images"][0]]
                    for name,field in (("creator","creator_group"),("original","original_asset_group"),("landing","original_landing_url")):
                        if provenance[field]!=asset[field]: raise ValueError("Photo manifest group mismatch")
                        _isolate(identities[name],provenance[field],identity,name)
                    if signatures[0]!=provenance["processed_sha256"] or signatures[0]!=asset["processed_sha256"]: raise ValueError("Photo processed hash mismatch")
                    raw_photo=confined(root/"data",f"raw/open_images_v7/images/{provenance['image_id']}.jpg")
                    if sha256_file(raw_photo)!=provenance["cvdf_file_sha256"] or provenance["cvdf_file_sha256"]!=asset["cvdf_file_sha256"]: raise ValueError("Original photo source hash mismatch")
                    counts["raw_label_fidelity"]+=1
                elif source=="mdprep_text_rules":
                    if text_answer(row)!=target: raise ValueError("Independent text-policy answer mismatch")
                    prefix={"choice":"text-policy-choice","noul":"text-policy-noul","score":"text-policy-score"}[row["question"]["type"]]
                    expected_group=stable_id(prefix,provenance["family"],row["audit"]["rule"],json.dumps(row["audit"]["facts"],sort_keys=True))
                    if row["group_id"]!=expected_group: raise ValueError("Generated text canonical fact identity mismatch")
                    counts["generated_rule_labels"]+=1
                elif source=="rule-variants-v1":
                    if recompute_rule_variant_label(row)!=row["target"]["label"]:
                        raise ValueError("Counterfactual rule-variant answer mismatch")
                    if row["split"] not in {"train","dev","calibration"}:
                        raise ValueError("Rule variants escaped their admitted splits")
                    rule_variant_splits[row["split"]]+=1
                    counts["rule_variant_labels"]+=1
                    counts["generated_rule_labels"]+=1
                elif source=="synthetic-visuals-v1":
                    relative=row["images"][0]
                    asset=indexes["visuals"][relative]
                    if asset["sha256"]!=signatures[0] or asset["group_id"]!=row["group_id"]: raise ValueError("Visual source reference mismatch")
                    if relative in failed_assets: raise ValueError("Visible extraction unresolved for this image")
                    if relative not in visual_cache:
                        visual_evidence[relative]={"status":"INCONCLUSIVE","media_sha256":signatures[0],"crops":[]}
                        try:
                            visual_cache[relative]=extract_visual(root/"data"/relative,provenance["scene_category"],row["split"]=="test_ood",visual_evidence[relative]["crops"])
                            visual_evidence[relative].update(status="EXTRACTED",facts=visual_cache[relative])
                        except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as error:
                            failed_assets.add(relative)
                            raise ValueError(f"Visible extraction unresolved: {error}") from error
                    visible=visual_cache[relative]
                    scene=asset["scene_facts"]
                    if row["group_id"]!=stable_id(source,scene): raise ValueError("Visual canonical render-fact identity mismatch")
                    if (provenance["scene_category"]!=scene["category"] or asset["scene_category"]!=scene["category"]
                            or provenance["scene_variant"]!=asset["scene_variant"]
                            or provenance["layout_family"]!=asset["layout_family"]
                            or scene["ood"] is not (row["split"]=="test_ood")):
                        raise ValueError("Visual category, variant, or layout declaration mismatch")
                    if visible["value"]!=scene["values"][asset["scene_variant"]] or visible["required_field_visible"]!=scene["required_field_visible"]: raise ValueError("Rendered metric disagrees with source scene")
                    if visual_answer(row,visible)!=target: raise ValueError("Independent pixel/rule answer mismatch")
                    counts["visible_generated_images"]+=1
                    counts["generated_rule_labels"]+=1
                elif source=="generated_temporal_videos":
                    relative=row["videos"][0]["path"]
                    asset=indexes["videos"][relative]
                    if asset["sha256"]!=signatures[0] or asset["group_id"]!=row["group_id"]: raise ValueError("Video source reference mismatch")
                    if row["group_id"]!=stable_id(source,row["split"],row["audit"]["scene_family"],row["audit"]["trajectory_index"]): raise ValueError("Temporal trajectory identity mismatch")
                    if relative not in frame_cache: frame_cache[relative]=extract_frames(root/"data"/relative,row["audit"]["scene_family"])
                    facts=frame_cache[relative]
                    canonical=reconcile_temporal_frames(row,facts)
                    orbit=[]
                    for seq in (canonical,list(reversed(canonical))):
                        orbit.extend(tuple(seq[i:]+seq[:i]) for i in range(4))
                    _isolate(identities["orbit"],stable_id(row["audit"]["scene_family"],min(orbit)),identity,"Temporal orbit")
                    if row["audit"]["scene_family"]=="color_sequence" and canonical==list(reversed(canonical)): raise ValueError("Self-reversing color sequence")
                    if video_answer(row,facts)!=target: raise ValueError("Independent decoded temporal answer mismatch")
                    counts["visible_generated_videos"]+=1
                    counts["generated_rule_labels"]+=1
                elif source=="hard-v1":
                    upstream_split=provenance.get("upstream_split")
                    if ((upstream_split=="train" and row["split"]!="train")
                            or (upstream_split=="development" and row["split"] not in {"dev","calibration"})
                            or upstream_split not in {"train","development"}):
                        raise ValueError("hard-v1 upstream split was admitted outside its allowed scope")
                    hardv1_splits[row["split"]]+=1
                    counts["hardv1_source_labels"]+=1
                    counts["generated_rule_labels"]+=1
                else:
                    raise ValueError(f"Unsupported source {source}")
                counts["records"]+=1
                receipts[row["id"]]=object_hash(row)
            except (ValueError,KeyError,TypeError,IndexError,StopIteration,OSError,RuntimeError,subprocess.CalledProcessError) as error:
                result["errors"].append({"id":row.get("id"),"error":str(error)})
        if hardv1_splits!={"train":7876,"dev":464,"calibration":464}:
            result["errors"].append({"error":f"hard-v1 split counts changed: {dict(hardv1_splits)}"})
        if rule_variant_splits!={"train":5400,"dev":350,"calibration":350}:
            result["errors"].append({"error":f"rule-variants split counts changed: {dict(rule_variant_splits)}"})
        intent_totals=collections.Counter()
        intent_sets=collections.defaultdict(set)
        for (split,key),distribution in bank_sets.items():
            for label,total in distribution.items():
                intent_totals[(split,label)]+=total
                intent_sets[(split,label)].add(key)
            if set(distribution)!=set(key) or len(set(distribution.values()))!=1:
                result["errors"].append({"split":split,"error":"Candidate-set oracle balance is not exactly 20%"})
        for (split,label),total in intent_totals.items():
            if total>=2 and len(intent_sets[(split,label)])<2:
                result["errors"].append({"split":split,"intent":label,"error":"BANKING77 distractor diversity is insufficient"})
        result["candidate_only_control"]=candidate_only_probe(rows)
        result["candidate_only_control"]["input_sha256"]=sha256_file(root/"data/parts/text.jsonl")
        result["candidate_only_control"]["per_set_oracle_accuracy"]=0.2
        if result["candidate_only_control"]["correct"]*5!=result["candidate_only_control"]["dev_questions"]:
            result["errors"].append({"error":"Candidate-only development control differs from 20%"})
        result["generated_shortcut_controls"]=generated_shortcut_check(root)
        if result["generated_shortcut_controls"]["structurally_fixed_numeric_rank_families"]:
            result["errors"].append({"error":"Generated numeric candidates retain a fixed correct rank"})
        result["checks"]={"schema_identity_isolation":{"status":"VERIFIED" if not result["errors"] else "NOT_VERIFIED"},"raw_label_fidelity":{"status":"VERIFIED" if counts["raw_label_fidelity"]==8270 else "NOT_VERIFIED","questions":counts["raw_label_fidelity"]},"generated_label_recomputation":{"status":"VERIFIED" if counts["generated_rule_labels"]==36634 else "NOT_VERIFIED","questions":counts["generated_rule_labels"]},"rule_variant_recomputation":{"status":"VERIFIED" if counts["rule_variant_labels"]==6100 else "NOT_VERIFIED","questions":counts["rule_variant_labels"]},"hardv1_source_conversion":{"status":"VERIFIED" if counts["hardv1_source_labels"]==8804 and hardv1_verification["status"]=="PASSED" else "NOT_VERIFIED","questions":counts["hardv1_source_labels"],"independent_receipt_sha256":sha256_file(hardv1_verifier_path)},"visible_generated_fact_extraction":{"status":"VERIFIED" if counts["visible_generated_images"]==15000 and counts["visible_generated_videos"]==3000 else "INCONCLUSIVE","image_questions":counts["visible_generated_images"],"video_questions":counts["visible_generated_videos"],"decoded_video_frames":4*len(frame_cache)}}
        result["questions_checked"]=counts["records"]
        evidence_path=root/"data/verification-evidence.json"
        write_json(evidence_path,{"visuals":visual_evidence,"videos":{path:{"media_sha256":media_hashes[path],"decoded_frame_facts":facts} for path,facts in frame_cache.items()},"human_review_performed":False})
        result["automated_evidence_sha256"]=sha256_file(evidence_path)
        if result["errors"] or counts["records"]!=44904 or any(check["status"]!="VERIFIED" for check in result["checks"].values()):
            write_json(root/"data/verification-report.json",result)
            return result
        source_receipt={"status":"VERIFIED","questions":8270,"source_manifest_hashes":source_hashes,"raw_source_hashes":raw_hashes,"method":"Pinned BANKING77 Git blobs and every selected utterance/label; Open Images verification rows, class names and image metadata; no independent photo truth adjudication."}
        verifier_files=("common.py","source_contracts.py","records.py","hardv1.py","rule_variants.py","assemble.py","shortcut_check.py","verify.py","verify_sources.py","verify_glyphs.py","verify_visuals.py","verify_video.py")
        verifier_hashes={f"src/mdprep/{name}":sha256_file(root/"src/mdprep"/name) for name in verifier_files}
        verifier_hashes["tools/verify_hardv1_conversion.py"]=sha256_file(root/"tools/verify_hardv1_conversion.py")
        hardv1_receipt_hashes={
            path.relative_to(root).as_posix():sha256_file(path)
            for path in (hardv1_verifier_path,hardv1_conversion_path,hardv1_admission_path)
        }
        rule_variant_receipt_hashes={
            path.relative_to(root).as_posix():sha256_file(path)
            for path in rule_variant_receipt_paths
        }
        contract={**result,"status":"VERIFIED","all_questions_checked":True,"scope":list(SPLITS),"prepared_hashes":prepared_hashes,"media_hashes":media_hashes,
                  "derived_hashes":{f"prepared/{name}.jsonl":sha256_file(root/"data/prepared"/f"{name}.jsonl") for name in ("pilot","diagnostic")},
                  "verifier_hashes":verifier_hashes,
                  "source_reconciliation":source_receipt,"source_receipt_sha256":object_hash(source_receipt),"source_receipt_hashes":{**source_hashes,**raw_hashes,**hardv1_receipt_hashes,**rule_variant_receipt_hashes},"row_receipts":receipts,
                  "qualification":"Experimental training only. Source-label fidelity and generated pixels are automated checks. Human review and independent photo visible truth remain unperformed."}
        contract["qualification_sha256"]=object_hash(contract)
        temporary=root/"data/verification-contract.json.partial"
        write_json(temporary,contract)
        validate_verification_contract(root,temporary)
        temporary.replace(root/"data/verification-contract.json")
        write_json(root/"data/verification-report.json",{key:value for key,value in contract.items() if key!="row_receipts"})
        return contract
    except (ValueError,KeyError,TypeError,IndexError,StopIteration,OSError,RuntimeError,subprocess.CalledProcessError) as error:
        result["errors"].append({"error":str(error)})
        write_json(root/"data/verification-report.json",result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    result=verify(parser.parse_args().root)
    print(json.dumps({key:value for key,value in result.items() if key not in {"row_receipts","media_hashes"}},indent=2))
    raise SystemExit(0 if result["status"]=="VERIFIED" else 1)


if __name__=="__main__":
    main()
