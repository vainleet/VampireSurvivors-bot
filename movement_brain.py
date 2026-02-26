import numpy as np
from config import KITE_MIN_DIST, KITE_MAX_DIST, KITE_TANGENT_FORCE, LOCAL_THREAT_LIMIT, RUNE_ATTRACT_FORCE, MONSTER_FORCE, PREDICT_FORCE
from utils import detect_fires

class MovementBrain:
    def nearest_enemy(self, player, enemies):
        if not enemies:
            return None
        return min(enemies, key=lambda e: np.hypot(e[0] - player[0], e[1] - player[1]))

    def kite_vector(self, player, enemies):
        nearest = self.nearest_enemy(player, enemies)
        if not nearest:
            return 0.0, 0.0

        ex, ey, conf = nearest

        dx = player[0] - ex
        dy = player[1] - ey
        dist = max(1, np.hypot(dx, dy))
        rx = dx / dist
        ry = dy / dist

        fx, fy = 0.0, 0.0

        if dist < KITE_MIN_DIST:
            f = (KITE_MIN_DIST - dist) * 4
            fx += rx * f
            fy += ry * f

        elif dist > KITE_MAX_DIST:
            f = (dist - KITE_MAX_DIST) * 2
            fx -= rx * f
            fy -= ry * f

        tx = -ry
        ty = rx
        fx += tx * KITE_TANGENT_FORCE / (dist + 40)
        fy += ty * KITE_TANGENT_FORCE / (dist + 40)

        return fx, fy

    def local_threat(self, point, enemies):
        threat = 0.0
        px, py = point
        for x, y, conf in enemies:
            d = max(1, np.hypot(x - px, y - py))
            threat += conf / (d * d)
        return threat

    def attractive_vector(self, player, target, base_force=1600.0):
        tx, ty, conf = target
        dx = tx - player[0]
        dy = ty - player[1]
        d = max(1, np.hypot(dx, dy))

        f = base_force * conf / (d + 25)
        return (dx / d * f, dy / d * f)

    def avoid_crowd(self, player, enemies, predicted):
        fx, fy = 0.0, 0.0
        for x, y, conf in enemies:
            dx = player[0] - x
            dy = player[1] - y
            d = max(1, np.hypot(dx, dy))
            f = MONSTER_FORCE * conf / (d + 30)
            fx += dx / d * f
            fy += dy / d * f
        for x, y in predicted:
            dx = player[0] - x
            dy = player[1] - y
            d = max(1, np.hypot(dx, dy))
            f = PREDICT_FORCE / (d + 30)
            fx += dx / d * f
            fy += dy / d * f

        return fx, fy

    def decide(self, player, enemies, exps, predicted, frame):
        fx, fy = 0.0, 0.0

        if enemies:
            kx, ky = self.kite_vector(player, enemies)
            fx += kx
            fy += ky

        attract_candidates = []

        for rune in exps:
            local_th = self.local_threat((rune[0], rune[1]), enemies)
            if local_th < LOCAL_THREAT_LIMIT:
                attract_candidates.append((rune, RUNE_ATTRACT_FORCE))

        fires = detect_fires(frame)
        for fire in fires:
            local_th = self.local_threat(fire, enemies)
            if local_th < LOCAL_THREAT_LIMIT:
                attract_candidates.append(((fire[0], fire[1], 0.9), 1200.0))

        if attract_candidates:
            attract_candidates.sort(
                key=lambda item: np.hypot(item[0][0] - player[0], item[0][1] - player[1])
            )
            target, force = attract_candidates[0]
            ax, ay = self.attractive_vector(player, target, force)
            fx = fx * 0.4 + ax
            fy = fy * 0.4 + ay

        ax, ay = self.avoid_crowd(player, enemies, predicted)
        fx += ax * 0.7
        fy += ay * 0.7

        mag = np.hypot(fx, fy)
        if mag > 0:
            fx /= mag
            fy /= mag

        return fx, fy