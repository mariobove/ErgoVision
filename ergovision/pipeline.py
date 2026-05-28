"""
ErgoVision pipeline — orchestrates dataset → pose estimation → scoring → output.
"""

import csv
import json
from pathlib import Path

import cv2

from .dataset import find_images, select_subset
from .pose_estimator import PoseEstimator
from .ergonomic_scoring import ErgonomicScorer
from .visualization import draw_skeleton, draw_risk_info, save_prediction
from .config import OUTPUT_DIR, VISUALIZATION_DIR, CSV_OUTPUT, JSON_OUTPUT


class ErgoPipeline:
    """End-to-end ErgoVision pipeline."""

    def __init__(self, model_name=None):
        self.pose_estimator = PoseEstimator(model_name=model_name)
        self.scorer = ErgonomicScorer()
        self.results = []

    def run(self, dataset_path, subset_size=50,
            save_visualizations=True, verbose=True):
        """
        Execute the full pipeline.

        Steps
        -----
        1. Walk *dataset_path* for images and select a subset.
        2. Run YOLOv8-pose inference.
        3. Compute ergonomic risk scores.
        4. Save visualisations, CSV, and JSON.

        Returns
        -------
        list[dict]  — one entry per image.
        """
        # ----- Step 1: dataset ------------------------------------------
        if verbose:
            print("=" * 60)
            print("ErgoVision — Ergonomic Risk Assessment Pipeline")
            print("=" * 60)
            print(f"\n[1/4] Inspecting dataset: {dataset_path}")

        all_images = find_images(dataset_path)
        images = select_subset(all_images, subset_size)

        if verbose:
            print(f"  Found {len(all_images)} images → subset of {len(images)}")

        # ----- Step 2: pose estimation ----------------------------------
        if verbose:
            print(f"\n[2/4] Running YOLOv8-pose on {len(images)} images...")

        detections = self.pose_estimator.estimate_batch(images, verbose=verbose)

        # ----- Step 3: scoring ------------------------------------------
        if verbose:
            print(f"\n[3/4] Computing ergonomic risk scores...")

        results = []
        for det in detections:
            entry = {
                'image_path': det['image_path'],
                'num_people': len(det['detections']),
                'people': [],
            }
            for idx, person in enumerate(det['detections']):
                score_result = self.scorer.score(person['keypoints'])
                person_data = {
                    'person': idx,
                    **score_result,
                }
                entry['people'].append(person_data)

                if verbose:
                    name = Path(det['image_path']).name
                    print(f"  {name}  person {idx}: "
                          f"{score_result['final_risk_class']} "
                          f"(score {score_result['final_score']})")

            results.append(entry)
        self.results = results

        # ----- Step 4: output -------------------------------------------
        if verbose:
            print(f"\n[4/4] Saving outputs...")

        if save_visualizations:
            self._save_visualizations(detections, results, verbose=verbose)

        self._save_csv(results, verbose=verbose)
        self._save_json(results, verbose=verbose)

        if verbose:
            self._print_summary(results)

        return results

    # -- private output helpers -------------------------------------------

    def _save_visualizations(self, detections, results, verbose=True):
        for det, res in zip(detections, results):
            image = cv2.imread(det['image_path'])
            if image is None:
                continue

            for person in det['detections']:
                image = draw_skeleton(
                    image, person['keypoints'], person['confidence']
                )

            if res['people']:
                p = res['people'][0]
                image = draw_risk_info(
                    image, p['final_risk_class'], p['final_score'],
                    p['explanation']
                )

            out_path = VISUALIZATION_DIR / Path(det['image_path']).name
            save_prediction(image, out_path)

            if verbose:
                print(f"  Saved: {out_path}")

    def _save_csv(self, results, verbose=True):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'image_path',
            'torso_angle', 'neck_angle', 'knee_angle',
            'shoulder_asymmetry', 'body_inclination',
            'torso_score', 'neck_score', 'knee_score',
            'shoulder_score', 'body_inclination_score',
            'final_score', 'final_risk_class',
            'explanation', 'unavailable_features',
        ]

        with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for res in results:
                for p in res['people']:
                    ps = p['partial_scores']
                    row = {
                        'image_path': res['image_path'],
                        'torso_angle':   ps.get('torso_angle', {}).get('value', ''),
                        'neck_angle':    ps.get('neck_angle', {}).get('value', ''),
                        'knee_angle':    ps.get('knee_angle', {}).get('value', ''),
                        'shoulder_asymmetry': ps.get('shoulder_asymmetry', {}).get('value', ''),
                        'body_inclination':   ps.get('body_inclination', {}).get('value', ''),
                        'torso_score':   ps.get('torso_angle', {}).get('score', ''),
                        'neck_score':    ps.get('neck_angle', {}).get('score', ''),
                        'knee_score':    ps.get('knee_angle', {}).get('score', ''),
                        'shoulder_score':    ps.get('shoulder_asymmetry', {}).get('score', ''),
                        'body_inclination_score': ps.get('body_inclination', {}).get('score', ''),
                        'final_score':       p['final_score'],
                        'final_risk_class':  p['final_risk_class'],
                        'explanation':       p['explanation'],
                        'unavailable_features': ', '.join(p['unavailable_features']),
                    }
                    writer.writerow(row)

        if verbose:
            print(f"  Saved CSV: {CSV_OUTPUT}")

    def _save_json(self, results, verbose=True):
        with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"  Saved JSON: {JSON_OUTPUT}")

    def _print_summary(self, results):
        total_people = sum(r['num_people'] for r in results)
        counts = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0}
        for r in results:
            for p in r['people']:
                counts[p['final_risk_class']] = \
                    counts.get(p['final_risk_class'], 0) + 1

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Images processed : {len(results)}")
        print(f"  People detected  : {total_people}")
        print(f"  Risk distribution:")
        for cls in ['Low Risk', 'Medium Risk', 'High Risk']:
            n = counts.get(cls, 0)
            pct = (n / max(total_people, 1)) * 100
            print(f"    {cls:15s}: {n:3d} ({pct:5.1f}%)")
        print(f"\n  Outputs → {OUTPUT_DIR.resolve()}")
        print("=" * 60)
