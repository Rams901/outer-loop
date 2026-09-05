"""Pixel cohort / one-user viewer. Consumes a TrueWorld pixel log.

  OMP_NUM_THREADS=1 PYTHONPATH=src python -m rl_opt.pixel_world

Requires pygame (pip install pygame). Not a city simulation: users sit in a
topic-sorted grid. Content does not float over the map. Feed tiles are the
ranked posts that user actually saw that day.
"""

from __future__ import annotations

import argparse
import sys

from rl_opt.pixel_log import record_pixel_logs

TOPIC_RGB = ((92, 201, 132), (90, 164, 255), (232, 186, 74), (206, 120, 220))
BG = (7, 10, 20)
SURF = (18, 24, 42)
TEXT = (238, 241, 248)
DIM = (132, 148, 180)
ROSE = (255, 84, 112)
BAIT = (255, 90, 110)
QUALITY = (90, 201, 150)
OON = (90, 110, 150)


def _need_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required:  pip install pygame") from exc
    return pygame


def _user_row(log: dict, day: int, user_id: int) -> dict:
    for rec in log["days"][day]:
        if rec["user_id"] == user_id:
            return rec
    raise KeyError(user_id)


def _tint(rgb: tuple[int, int, int], sat: float, returned: bool) -> tuple[int, int, int]:
    if not returned:
        return (48, 52, 64)
    s = 0.25 + 0.75 * max(0.0, min(1.0, sat))
    return tuple(int(c * s) for c in rgb)


def run_viewer(payload: dict) -> int:
    pygame = _need_pygame()
    pygame.init()
    pygame.display.set_caption("rl_opt pixel world — 21-day feed (not a city)")
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    small = pygame.font.Font(None, 18)
    tiny = pygame.font.Font(None, 15)

    config = "default"
    day = 0
    playing = False
    acc = 0.0
    users = payload["configs"]["default"]["users"]
    n_days = len(payload["configs"]["default"]["days"])
    selected = users[0]["user_id"]
    ids_by_topic = sorted(users, key=lambda u: (u["topic"], u["user_id"]))
    n = len(ids_by_topic)
    cols = 12
    rows = (n + cols - 1) // cols
    cell = 28
    grid_origin = (24, 88)

    def log() -> dict:
        return payload["configs"][config]

    running = True
    while running:
        dt = clock.tick(30) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    day = min(n_days - 1, day + 1)
                    playing = False
                elif event.key == pygame.K_LEFT:
                    day = max(0, day - 1)
                    playing = False
                elif event.key == pygame.K_TAB:
                    config = "engagement" if config == "default" else "default"
                elif event.key == pygame.K_r:
                    day = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                gx = (mx - grid_origin[0]) // cell
                gy = (my - grid_origin[1]) // cell
                if 0 <= gx < cols and 0 <= gy < rows:
                    idx = gy * cols + gx
                    if idx < n:
                        selected = ids_by_topic[idx]["user_id"]

        if playing:
            acc += dt
            if acc >= 0.55:
                acc = 0.0
                if day < n_days - 1:
                    day += 1
                else:
                    playing = False

        L = log()
        cohort = L["cohort"]
        rec = _user_row(L, day, selected)
        screen.fill(BG)

        title = font.render(
            f"day {day}/{n_days - 1}   config={config}   "
            f"cohort return {cohort['daily_return'][day]:.2f}   "
            f"bait {cohort['daily_bait_share'][day]:.2f}   "
            f"engage {cohort['daily_engagement'][day]:.3f}   "
            f"sat {cohort['daily_satisfaction'][day]:.2f}",
            True,
            TEXT,
        )
        screen.blit(title, (24, 16))
        hint = small.render(
            "click a user   ← → day   space play   tab default/engagement trap   "
            "tiles: rose=bait  green=in-interest  slate=OON   gray cell=churned",
            True,
            DIM,
        )
        screen.blit(hint, (24, 44))
        vline = small.render(
            f"full-horizon V={cohort['value']:.3f}  2-day proxy={cohort['proxy']:.3f}   "
            "in-interest ≈ followed-ish (relevance); OON ≈ non-follower. No follow graph in this world.",
            True,
            DIM,
        )
        screen.blit(vline, (24, 64))

        for i, u in enumerate(ids_by_topic):
            gx, gy = i % cols, i // cols
            row = _user_row(L, day, u["user_id"])
            color = _tint(TOPIC_RGB[u["topic"] % 4], row["satisfaction"], row["returned"])
            x = grid_origin[0] + gx * cell
            y = grid_origin[1] + gy * cell
            pygame.draw.rect(screen, color, (x + 1, y + 1, cell - 2, cell - 2), border_radius=3)
            if row["returned"] and row["bait_share"] >= 0.4:
                pygame.draw.rect(screen, ROSE, (x + 3, y + 3, 6, 6))
            if u["user_id"] == selected:
                pygame.draw.rect(screen, TEXT, (x, y, cell, cell), width=2, border_radius=4)

        panel = pygame.Rect(24 + cols * cell + 36, 88, 860, 600)
        pygame.draw.rect(screen, SURF, panel, border_radius=10)
        px, py = panel.x + 18, panel.y + 16
        head = font.render(
            f"user {selected}   topic {rec['topic']}   "
            f"{'OPENED the app' if rec['returned'] else 'DID NOT RETURN'}",
            True,
            TEXT,
        )
        screen.blit(head, (px, py))
        stats = small.render(
            f"satisfaction {rec['satisfaction']:.3f}   Δitem {rec['delta_s']:+.3f}   "
            f"likes {rec['n_likes']}  dwells {rec['n_dwells']}  hides {rec['n_hides']}   "
            f"bait {rec['n_bait']}/12  in-interest {rec['n_in_interest']}  OON {rec['n_oon']}   "
            f"engagement {rec['engagement']:.2f}",
            True,
            DIM,
        )
        screen.blit(stats, (px, py + 28))

        sat_w = int(400 * rec["satisfaction"])
        pygame.draw.rect(screen, (28, 34, 52), (px, py + 56, 400, 10), border_radius=4)
        pygame.draw.rect(screen, (124, 92, 255), (px, py + 56, sat_w, 10), border_radius=4)
        screen.blit(tiny.render("retention tendency (hidden sat → p(return tomorrow))", True, DIM), (px, py + 70))

        feed = rec["feed"]
        if not rec["returned"]:
            screen.blit(font.render("No feed — user churned this day.", True, ROSE), (px, py + 110))
        elif not feed:
            screen.blit(font.render("No feed recorded.", True, DIM), (px, py + 110))
        else:
            screen.blit(small.render("today's ranked feed  (slot 0 = top)", True, DIM), (px, py + 96))
            for i, item in enumerate(feed):
                fy = py + 118 + i * 36
                if item["bait"]:
                    fill = BAIT
                    kind = "BAIT"
                elif item["in_interest"]:
                    fill = QUALITY
                    kind = "in-interest"
                else:
                    fill = OON
                    kind = "OON"
                pygame.draw.rect(screen, fill, (px, fy, 22, 28), border_radius=3)
                acts = []
                if item["like"]:
                    acts.append("like")
                if item["dwell"]:
                    acts.append("dwell")
                if item["hide"]:
                    acts.append("hide")
                act = "  ".join(acts) if acts else "—"
                line = small.render(
                    f"#{item['slot']:02d}  post {item['post_id']:<4}  {kind:<12}  "
                    f"q={item['quality']:.2f}  rel={item['relevance']:.2f}  {act}",
                    True,
                    TEXT,
                )
                screen.blit(line, (px + 32, fy + 6))

        pygame.display.flip()

    pygame.quit()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="21-day pixel viewer over a real TrueWorld rollout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch", type=int, default=96)
    args = parser.parse_args()
    print("recording paired default vs engagement rollouts …", flush=True)
    payload = record_pixel_logs(world_seed=args.seed, n_watch=args.watch)
    return run_viewer(payload)


if __name__ == "__main__":
    sys.exit(main())
