"""Behavioral regressions for corpus boundaries and observed label defects."""

import copy
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from mdprep.common import choice, record, sha256_file, write_json, write_jsonl
from mdprep.records import parse_record, parse_request, validate_record
from mdprep.text import SourceExample, _intent_blocks
from mdprep.verify import _isolate, candidate_only_probe, object_hash, validate_verification_contract, verify
from mdprep.verify_video import center, temporal_answer
from mdprep.verify_visuals import extract_visual, integer
from mdprep.videos import _collision_trajectories, _color_trajectories, _sequence_orbit, _transform_collision_points


def example(split="train", variant=0):
    return record(split=split,source="mdprep_text_rules",group_id="group",family="id_transfer_release_constraints",modality="text",state="Observed evidence",question=choice("Choose an action",{"allow":"Allow","decline":"Decline"}),label="decline",provenance={"license":"Project-created"},variant=variant)


class RecordBoundaryTests(unittest.TestCase):
    def test_neutral_keys_restore_original_response_keys(self):
        row=example()
        parsed=parse_record(row)
        self.assertEqual(list(row["question"]["criteria"]),["option_0","option_1"])
        self.assertEqual(parsed.input.candidates[parsed.target_index].key,"decline")
        self.assertEqual(parsed.input.candidates[parsed.target_index].description,"Decline")
        self.assertEqual(parsed.origin,"generated_rule")
        self.assertEqual(parsed.ood_axis,"none")
        self.assertEqual(row["provenance"]["variant"],0)

    def test_variant_roundtrip_and_tampering(self):
        row=json.loads(json.dumps(example(variant=(3,2))))
        self.assertEqual(parse_record(row).input.id,row["id"])
        row["provenance"]["variant"]=[3,1]
        with self.assertRaisesRegex(ValueError,"identity"):
            validate_record(row)

    def test_strict_schema_rejects_extra_labels_and_metadata_mismatch(self):
        row=example()
        row["extra"]=True
        with self.assertRaises(ValueError): validate_record(row)
        row=example()
        row["audit"]["original_target_key"]="allow"
        with self.assertRaisesRegex(ValueError,"target"):
            validate_record(row)
        request={"id":"r","state":"","kind":"choice","instructions":"Choose","candidates":[{"key":"a","description":"A"},{"key":"b","description":"B"}],"images":[],"videos":[{"path":"../escape.mp4","fps":2,"duration_s":2,"frame_count":4}]}
        with self.assertRaisesRegex(ValueError,"confined"): parse_request(request)
        request["videos"][0]["path"]="media/clip.mp4"
        request["videos"][0]["fps"]=float("nan")
        with self.assertRaisesRegex(ValueError,"fps"): parse_request(request)

    def test_same_bytes_different_group_are_rejected_within_split(self):
        mapping={}
        _isolate(mapping,"sha",("train","first"),"Media")
        _isolate(mapping,"sha",("train","first"),"Media")
        self.assertEqual(mapping,{"sha":("train","first")})
        with self.assertRaisesRegex(ValueError,"different source groups"):
            _isolate(mapping,"sha",("train","second"),"Media")


class IntentBlockTests(unittest.TestCase):
    def test_shared_sets_preserve_source_examples_and_balance_each_intent(self):
        labels=("card_arrival","cash_withdrawal_amount","card_delivery_estimate","cash_withdrawal_charge","card_payment_fee_charged","card_swallowed")
        examples=[SourceExample(f"message {label} {i}",label,100*i+j,"normalized",f"cluster-{i}-{j}",1) for i in range(5) for j,label in enumerate(labels)]
        result=_intent_blocks(examples,"train")
        self.assertEqual(Counter(e.cluster_id for e,c in result),Counter(e.cluster_id for e in examples))
        by_set={}
        for item,candidates in result:
            self.assertEqual(len(candidates),5)
            by_set.setdefault(tuple(candidates),Counter())[item.category]+=1
        for candidates,counts in by_set.items():
            self.assertEqual(set(counts),set(candidates))
            self.assertEqual(len(set(counts.values())),1)
        self.assertEqual(len(result),30)

    def test_infeasible_queues_do_not_relabel_or_silently_drop(self):
        examples=[SourceExample(str(i),"dominant" if i<6 else f"label-{i}",i,str(i),str(i),1) for i in range(10)]
        with self.assertRaisesRegex(ValueError,"cannot form"):
            _intent_blocks(examples,"train")

    def test_description_only_probe_is_chance_on_balanced_sets(self):
        rows=[]
        for split in ("train","dev"):
            for target in range(5):
                rows.append({"split":split,"provenance":{"source":"banking77"},"state":"not used","question":{"criteria":{str(i):f"Class {i}" for i in range(5)}},"target":{"label":str(target)}})
        result=candidate_only_probe(rows)
        self.assertEqual((result["correct"],result["dev_questions"],result["accuracy"]),(1,5,0.2))
        self.assertEqual(result["known_sets_in_dev"],5)
        self.assertFalse(result["final_test_used"])


class ObservableGeometryTests(unittest.TestCase):
    def test_triangle_center_near_top_is_observed_without_clipping(self):
        image=Image.new("RGB",(256,192),"#F4F1E8")
        draw=ImageDraw.Draw(image)
        vertices=((90,14),(74,45),(106,45))
        draw.polygon(vertices,fill="#E0B83D",outline="#25231F")
        draw.line((*vertices,vertices[0]),fill="#25231F",width=3)
        self.assertEqual(center(image,"yellow",shape="triangle"),(90,31))

    def test_integer_rotations_preserve_equal_distances(self):
        actual=_transform_collision_points(((100,96),(112,96)),180,3,-2)
        self.assertEqual(actual,((159,94),(147,94)))
        with self.assertRaisesRegex(ValueError,"integer"):
            _transform_collision_points(((100,96),),7,0,0)
        for amber,violet in _collision_trajectories():
            distances=[math.dist(a,b) for a,b in zip(amber,violet)]
            for a,b in zip(distances,distances[1:]):
                self.assertTrue(a==b or abs(a-b)>=1)
        facts=[{"amber_center":a,"violet_center":b} for a,b in zip(((100,96),(112,96),(126,96),(170,96)),((128,96),(140,96),(130,96),(86,96)))]
        self.assertEqual(temporal_answer("collision.early_distance_trend",facts),"includes_no_change")
        self.assertEqual(temporal_answer("collision.contact_count",facts),"3")

    def test_palindromic_or_cyclic_duplicate_colors_are_not_allocated(self):
        trajectories=_color_trajectories()
        self.assertGreater(len(trajectories),80)
        for colors in trajectories:
            self.assertEqual(len(set(_sequence_orbit(colors))),8)
            self.assertNotEqual(colors,tuple(reversed(colors)))
        facts=[{"focus_color":c} for c in ("red","blue","red","green")]
        self.assertEqual(temporal_answer("color.return",facts),"true")
        self.assertEqual(temporal_answer("color.change_count",facts),"3")
        self.assertEqual(temporal_answer("color.penultimate",facts),"red")

    def test_subpixel_temporal_trend_stays_unresolved(self):
        facts=[{"amber_center":(0,0),"violet_center":p} for p in ((28,1),(28,2),(4,0),(84,0))]
        with self.assertRaisesRegex(ValueError,"Subpixel"):
            temporal_answer("collision.early_distance_trend",facts)

    def test_spatial_pixels_include_triangle_above_y120(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"scene.png"
            image=Image.new("RGB",(450,450),"white")
            draw=ImageDraw.Draw(image)
            draw.polygon(((58,116),(40,152),(76,152)),fill="#c84d42")
            draw.polygon(((138,180),(120,216),(156,216)),fill="#c84d42")
            draw.ellipse((50,330,90,370),fill="#3578b8")
            draw.rectangle((200,330,236,366),fill="#252a2c")
            image.save(path)
            result=extract_visual(path,"spatial",False)
            self.assertEqual((result["required_field_visible"],result["value"],result["lookup"],result["predicate"]),
                             (False,2,"2",True))

    def test_ambiguous_ocr_does_not_pass(self):
        self.assertEqual(integer("$42.00"),42)
        for value in ("", "42 43", "42.50"):
            with self.assertRaises(ValueError): integer(value)



class QualificationTests(unittest.TestCase):
    def test_failed_verification_does_not_overwrite_prior_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            destination=root/"data/verification-contract.json"
            write_json(destination,{"prior":"untouched"})
            before=destination.read_bytes()
            result=verify(root)
            self.assertEqual(result["status"],"NOT_VERIFIED")
            self.assertGreater(len(result["errors"]),0)
            self.assertEqual(destination.read_bytes(),before)

    def test_projected_receipt_needs_no_raw_sources_and_binds_full_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            row=example()
            path=root/"data/prepared/train.jsonl"
            write_jsonl(path,[row])
            receipt={"status":"VERIFIED","questions":8270}
            contract={"status":"VERIFIED","all_questions_checked":True,"questions_checked":1,"scope":["train"],"prepared_hashes":{"train":sha256_file(path)},"media_hashes":{},"row_receipts":{row["id"]:object_hash(row)},"source_reconciliation":receipt,"source_receipt_sha256":object_hash(receipt),"human_review":{"status":"NOT_VERIFIED"},"qualification":{"contract_sha256":"upstream-receipt","questions_checked":30000,"scope":["train","dev"]}}
            contract["qualification_sha256"]=object_hash(contract)
            destination=root/"data/verification-contract.json"
            write_json(destination,contract)
            self.assertEqual(validate_verification_contract(root)["questions_checked"],1)
            altered=copy.deepcopy(row)
            altered["state"]="Changed evidence with the same ID"
            write_jsonl(root/"data/prepared/pilot.jsonl",[altered])
            contract["derived_hashes"]={"prepared/pilot.jsonl":sha256_file(root/"data/prepared/pilot.jsonl")}
            contract.pop("qualification_sha256")
            contract["qualification_sha256"]=object_hash(contract)
            write_json(destination,contract)
            with self.assertRaisesRegex(ValueError,"changed or unqualified"):
                validate_verification_contract(root)


if __name__=="__main__":
    unittest.main()
