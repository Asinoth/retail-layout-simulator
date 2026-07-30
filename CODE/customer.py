import time
import math
import numpy as np
from collections import defaultdict
import random  # not always used but sometimes handy


# TODO: maybe refactor this whole class later
from customer_pathfinding import PathfindingMixin
from retail_literature import (
    IMPULSE_CHECKOUT_RADIUS_M, IMPULSE_DECAY_PER_ITEM,
    IMPULSE_BASE_PROPENSITY, IMPULSE_BROWSER_BONUS,
    IMPULSE_LONG_DWELL_BONUS, IMPULSE_LONG_DWELL_SECS,
)
class Customer(PathfindingMixin):
   """Customer in shop sim. Handles movement, shopping, states etc."""


   def __init__(self, customer_id, start_pos, shop_items, shop_dimensions, door_position=None, door_side=None, floor=1):
            self.id= customer_id
            self.position = list(start_pos)
            self.target_position = None
            self.speed = np.random.uniform(0.8, 1.5)  #m/s
            self.shopping_list=[]
            self.visited_items = set()
            self.attempted_items = set()
            self.abandoned_items = set()
            self._last_progress_pos = None
            self._last_progress_time = 0.0
            self.state = 'entering'
            self.dwell_time = 0
            self.start_shop_time = time.time()
            self.total_time_in_shop=0
            self.path = []
            self.path_index = 0
            self.shop_dimensions = shop_dimensions
            self.last_update_time= time.time()
            self.has_checked_out = False
            self._checkout_arrived = False
            self.checkout_lane = None   # which checkout lane this customer queues at
            self.abandon_probability = np.random.uniform(0.01, 0.05)
            self.abandon_cart = np.random.random() < self.abandon_probability
            self.entering_time = np.random.uniform(0.25,0.5)
            self.state_start_time = time.time()

            # impulse stuff
            self.impulse_items = []
            self.has_made_impulse_purchase = False
            self.impulse_purchase_attempts=0
            self.max_impulse_items = np.random.randint(1,4)
        
            self.door_position = door_position
            self.door_side = door_side

            # multi-floor state
            self.floor = floor
            self.target_floor = floor
            self._pending_connector = None
            self._pending_route_type = None
            self._pending_route_item = None
            self._pending_route_dest = None
        
            # analytics tracking
            self.basket_value = 0.0
            self.zone_dwell_times = defaultdict(float)
            self.visited_zones = set()
            self.purchase_intent= np.random.uniform(0.7, 0.95)
            self.customer_type = np.random.choice(['quick', 'browser','thorough'], p=[0.3, 0.4, 0.3])
            self.loyalty_level = np.random.choice(["new", "regular", "vip"], p=[0.5, 0.35, 0.15])
            self.movement_path = []
        
            self._generate_shopping_list(shop_items)
            self.size=0.15
    


   def get_color(self):
        """Get color based on state"""
        state_colors = {
            'entering': '#00ff00',
            "moving": '#0000ff',
            'shopping': "#ffa500",
            'checking_out': '#ff0000',
            "exiting": '#800080'
        }
        return state_colors.get(self.state,'#0000ff')
    
   @property
   def color(self):
        return self.get_color()



   def _generate_shopping_list(self, shop_items):
        """Generate random shopping list. Skip impulse/checkout/wc"""
        if not shop_items:
            return
        
        # filter out special items
        regular_items = {}
        for nm, data in shop_items.items():
            cat = data.get('category')
            if cat != 'Impulse' and nm not in ('Checkout','WC'):
                regular_items[nm] = data
        
        if len(regular_items) == 0:
            return
        
        # FIXME: hardcoded patterns - should be config
        patterns = {
                'quick': (2, 4, 15.0, 25.0),
                'browser': (4,7,35.0, 60.0),
                'thorough': (6,12,60.0,120.0)
        }
        
        cust_type = getattr(self, 'customer_type', "browser")
        min_items, max_items, min_val, max_val = patterns[cust_type]
        
        num = min(np.random.randint(min_items,max_items + 1), len(regular_items))
        self.shopping_list = np.random.choice(list(regular_items.keys()), 
                    size=num, replace=False).tolist()
        
        # calc basket value
        val = 0.0
        for itm in self.shopping_list:
            val = val + shop_items[itm].get('price', 0.0)
        self.basket_value = val
        
        self.impulse_probability = IMPULSE_BASE_PROPENSITY[cust_type]
        
        # wc prob
        self.wc_probability = {'quick': 0.15, 'browser': 0.35, 'thorough': 0.50}[cust_type]
        self.needs_wc = np.random.random() < self.wc_probability
        self.visited_wc = False




    
   def _initialize_path(self, walls):
        """Plan waypoints avoiding walls"""
        self.path = self._compute_path(self.position,self.target_position, walls)
        self.path_index = 0
        if len(self.path) > 0:
            self.target_position = self.path[0]







   def update(self, dt, shop_items, walls, other_customers):
            """Main update loop - handles all states"""
            current_time = time.time()
            #print(f"updating customer {self.id}, state={self.state}")

            if self.state == "entering":
                time_in = current_time - self.state_start_time
                if time_in >= self.entering_time:
                    if len(self.shopping_list) == 0:
                        self.current_target_type = 'checkout'
                        self._set_checkout_target(walls)
                        self._change_state("moving")
                    else:
                        self._choose_next_target(shop_items, walls)
                        self._change_state('moving')
                return

            elif self.state == "moving":
                if self.target_position is not None:
                    prev_x, prev_y = self.position[0], self.position[1]
                    self._move_towards_target(dt, walls)

                    # Track real progress; if we haven't moved more than
                    # 0.05m for 3+ seconds while still in 'moving', we are
                    # stuck on a wall or have no valid path. Drop the
                    # target so we don't freeze visually.
                    moved = (self.position[0] - prev_x) ** 2 + (self.position[1] - prev_y) ** 2
                    if moved > 0.0025:
                        self._last_progress_pos = (self.position[0], self.position[1])
                        self._last_progress_time = current_time
                    elif self._last_progress_pos is None:
                        self._last_progress_pos = (self.position[0], self.position[1])
                        self._last_progress_time = current_time
                    elif (current_time - self._last_progress_time) > 3.0:
                        # We have been stuck for >3s. Abandon current target
                        # and pick a new one.
                        if (getattr(self, 'current_target_item', None)
                                and getattr(self, 'current_target_type', None) in ('item', 'impulse')):
                            self.abandoned_items.add(self.current_target_item)
                            self.attempted_items.add(self.current_target_item)
                            try:
                                self.shopping_list = [
                                    i for i in self.shopping_list
                                    if i != self.current_target_item
                                ]
                            except Exception:
                                pass
                        self.current_target_item = None
                        # Re-pick a target so the customer keeps moving.
                        sim_ref = getattr(self, '_simulation_ref', None)
                        all_items = (sim_ref.shop.all_items_across_floors()
                                     if sim_ref is not None
                                     else shop_items)
                        if self.has_checked_out:
                            self.current_target_type = 'exit'
                            self._set_exit_target(walls)
                            # Only transition to 'exiting' once we're actually
                            # on floor 1 walking to the door. If _set_exit_target
                            # routed us via a connector (target_type='connector')
                            # we must stay in 'moving' so the connector arrival
                            # handler can fire and teleport us to floor 1.
                            if getattr(self, 'current_target_type', None) == 'exit':
                                self._change_state('exiting')
                        elif not any(i not in self.abandoned_items
                                     and i not in self.visited_items
                                     for i in self.shopping_list):
                            self.current_target_type = 'checkout'
                            self._set_checkout_target(walls)
                        else:
                            self._choose_next_target(all_items, walls)
                        self._last_progress_pos = (self.position[0], self.position[1])
                        self._last_progress_time = current_time
                        return

                    if self._reached_target() == True:
                        if hasattr(self, 'path') and self.path_index + 1 < len(self.path):
                            self.path_index = self.path_index + 1
                            self.target_position = self.path[self.path_index]
                            return

                        ttype = getattr(self, 'current_target_type', None)

                        # LOS guard for item/impulse/checkout/wc: only declare
                        # arrival if we are physically inside the target's
                        # bounding box (not just within 0.5m through a wall).
                        if ttype in ('item', 'impulse'):
                            sim_ref = getattr(self, '_simulation_ref', None)
                            data = None
                            if sim_ref is not None and self.current_target_item:
                                try:
                                    all_items = sim_ref.shop.all_items_across_floors()
                                except Exception:
                                    all_items = {}
                                data = all_items.get(self.current_target_item)
                            if data is not None:
                                ix, iy = data['position']
                                iw, ih = data['size']
                                margin = self.size + 0.05
                                if not (ix - margin <= self.position[0] <= ix + iw + margin
                                        and iy - margin <= self.position[1] <= iy + ih + margin):
                                    return  # keep walking
                        elif ttype in ('checkout', 'wc'):
                            if ttype == 'checkout':
                                key = getattr(self, 'checkout_lane', None) or 'Checkout'
                                w = walls.get(key) or walls.get('Checkout')
                            else:
                                key = 'WC'
                                w = walls.get(key)
                            if w is not None:
                                wx, wy = w['position']
                                ww, wh = w['size']
                                margin = self.size + 0.05
                                if not (wx - margin <= self.position[0] <= wx + ww + margin
                                        and wy - margin <= self.position[1] <= wy + wh + margin):
                                    return  # keep walking
                        if ttype == 'checkout':
                            self._change_state("checking_out")
                            self.dwell_time = np.random.uniform(3, 8)
                            return

                        if ttype == 'impulse':
                            self.impulse_items.append(self.current_target_item)
                            self._change_state("shopping")
                            self.dwell_time = np.random.uniform(2, 5)
                            return

                        if ttype == 'wc':
                            self._change_state('shopping')
                            self.dwell_time = np.random.uniform(2, 8)
                            self.visited_wc = True
                            if hasattr(self, '_simulation_ref'):
                                self._simulation_ref.analytics['area_visits']['WC'] += 1
                            return

                        # Connector arrival -- handle FIRST so we don't accidentally enter 'shopping'.
                        if ttype == 'connector':
                            self._handle_connector_arrival()
                            return

                        # Exit arrival from 'moving' state (stuck-recovery can
                        # set target_type='exit' without changing state). Hand
                        # off to 'exiting' so the simulation_loop can despawn
                        # the customer at the door.
                        if ttype == 'exit':
                            self._change_state('exiting')
                            return

                        # Default: arrived at an item shelf -> shop here.
                        self._change_state('shopping')
                        self.dwell_time = np.random.uniform(5, 15)
                        return
                return
                        

            elif self.state == 'shopping':
                self.dwell_time = self.dwell_time - dt
                if not (self.dwell_time > 0):
                    # handle impulse queue
                    if hasattr(self,'impulse_queue'):
                        if len(self.impulse_queue) > 0:
                            next_item = self.impulse_queue.pop(0)
                            self.current_target_item = next_item
                            self.current_target_type = 'impulse'
                            item = shop_items[next_item]
                            self.target_position = [
                                item['position'][0] + item['size'][0]/2,
                                item['position'][1] + item['size'][1]/2
                            ]
                            self._change_state("moving")
                            return
                        else:
                            del self.impulse_queue
                            self.current_target_type = 'checkout'
                            self._set_checkout_target(walls)
                            self._change_state("moving")
                            return

                    # after wc
                    if getattr(self, 'current_target_type', None) == 'wc':
                        self.current_target_type = None
                        if len(self.shopping_list) == 0:
                            self.current_target_type = 'checkout'
                            self._set_checkout_target(walls)
                            self._change_state("moving")
                        else:
                            self._choose_next_target(shop_items, walls)
                            self._change_state("moving")
                        return

                    if self.current_target_item:
                        self.visited_items.add(self.current_target_item)

                    # wc logic
                    if (self.needs_wc and self.visited_wc == False and
                        len(self.visited_items) >= len(self.shopping_list)//2):
                        self.current_target_type = 'wc'
                        self._set_wc_target(walls)
                        self._change_state("moving")
                        return

                    if len(self.visited_items) >= len(self.shopping_list):
                        if self.abandon_cart == True:
                            self._change_state("exiting")
                            self._set_exit_target(walls)
                        else:
                            self.current_target_type = 'checkout'
                            self._set_checkout_target(walls)
                            self._change_state("moving")
                        return

                    self._choose_next_target(shop_items, walls)
                    self._change_state("moving")
                return

            elif self.state == 'checking_out':
                if self.has_checked_out == False:
                    self.dwell_time = self.dwell_time - dt
                    if not (self.dwell_time > 0):
                        self.has_checked_out = True
                        self.checkout_end_time = time.time()
                        self._attempt_impulse_purchases(shop_items, walls)
                        if getattr(self, 'impulse_queue', None):
                            return
                    return
          
                if self.needs_wc and self.visited_wc == False:
                    self.current_target_type = 'wc'
                    self._set_wc_target(walls)
                    self._change_state("moving")
                else:
                    self._change_state("exiting")
                    self._set_exit_target(walls)
                return

            elif self.state == "exiting":
                if self.target_position:
                    prev_x, prev_y = self.position[0], self.position[1]
                    if hasattr(self, 'path') and self._reached_target():
                        if self.path_index + 1 < len(self.path):
                            self.path_index = self.path_index + 1
                            self.target_position = self.path[self.path_index]
                            self._move_towards_target(dt, walls)
                            return
                        # End of path -- if we were routed through a connector
                        # (e.g. abandon-cart from a non-ground floor), perform
                        # the floor teleport here too. Otherwise the customer
                        # would freeze at the stair center because the
                        # 'moving'-state connector handler never fires.
                        if getattr(self, 'current_target_type', None) == 'connector':
                            self._handle_connector_arrival()
                            return
                    self._move_towards_target(dt, walls)

                    # Stuck-detection: if we haven't made progress toward the
                    # exit for >5 s, hard-teleport to floor 1 + door so the
                    # simulation_loop can despawn us. Without this a customer
                    # walled off from the door (e.g. user moved walls after
                    # spawn) hangs on screen forever.
                    moved = (self.position[0] - prev_x) ** 2 + (self.position[1] - prev_y) ** 2
                    if moved > 0.0025:
                        self._last_progress_pos = (self.position[0], self.position[1])
                        self._last_progress_time = current_time
                    elif self._last_progress_pos is None:
                        self._last_progress_pos = (self.position[0], self.position[1])
                        self._last_progress_time = current_time
                    elif (current_time - self._last_progress_time) > 5.0:
                        sim = getattr(self, '_simulation_ref', None)
                        if sim is not None and sim.door_position is not None:
                            self.floor = 1
                            self.position = list(sim.door_position)
                            self.target_position = list(sim.door_position)
                            self.current_target_type = 'exit'
                            self._pending_connector = None
                            self._last_progress_time = current_time
                return




   def _attempt_impulse_purchases(self, shop_items, walls):
        """Check for impulse items near checkout"""
        if self.has_made_impulse_purchase == True:
            return

        impulse_candidates = []
        cx_co, cy_co = self.target_position

        for item_name, data in shop_items.items():
                if data.get('category') != 'Impulse':
                    continue
                if data.get('floor', self.floor) != self.floor:
                    continue
                w, h = data['size']
                cx = data['position'][0] + w/2
                cy = data['position'][1] + h/2
                dist = math.hypot(cx_co - cx, cy_co - cy)
                if dist <= IMPULSE_CHECKOUT_RADIUS_M:
                    impulse_candidates.append(item_name)

        if len(impulse_candidates) == 0:
            self.has_made_impulse_purchase = True
            return

        chance = self.impulse_probability
        if self.customer_type == 'browser':
                chance = chance + IMPULSE_BROWSER_BONUS
        if self.total_time_in_shop > IMPULSE_LONG_DWELL_SECS:
                chance = chance + IMPULSE_LONG_DWELL_BONUS

        self.impulse_items_to_visit = []
        cnt = 0
        while (cnt < self.max_impulse_items
                   and len(impulse_candidates) > 0
                   and np.random.random() < chance):
                choice = np.random.choice(impulse_candidates)
                impulse_candidates.remove(choice)
                self.impulse_items_to_visit.append(choice)
                cnt = cnt + 1
                chance = chance * IMPULSE_DECAY_PER_ITEM

        self.has_made_impulse_purchase = True
        self.impulse_purchase_attempts = cnt

        if len(self.impulse_items_to_visit) > 0:
                    self.impulse_queue = self.impulse_items_to_visit.copy()
                    self._set_next_impulse_target(shop_items, walls)




   def _set_next_impulse_target(self, shop_items, walls):
        """Pop next impulse item and set as target"""
        if not getattr(self, 'impulse_queue', None):
            return
        next_item = self.impulse_queue.pop(0)
        self.current_target_item = next_item
        self.current_target_type = 'impulse'
        data = shop_items[next_item]
        item_floor = data.get('floor', self.floor)
        self.target_floor = item_floor
        if item_floor != self.floor:
            self._route_to_floor(item_floor, walls)
            self._change_state("moving")
            return
        self.target_position = [
            data['position'][0] + data['size'][0]/2,
            data['position'][1] + data['size'][1]/2
        ]
        self._change_state("moving")
        self._initialize_path(walls)




   def _set_wc_target(self, walls):
        """Set target to WC. If WC not on current floor, try to find it
        on another floor via connector."""
        if 'WC' in walls:
            wc = walls['WC']
            pos = wc['position']
            sz = wc['size']
            self.target_position = [
                pos[0] + sz[0]/2,
                pos[1] + sz[1]/2
            ]
            self.current_target_type = 'wc'
            self.current_target_item = None
            self._initialize_path(walls)
        else:
            # WC might be on another floor -- check all floors
            sim = getattr(self, '_simulation_ref', None)
            reachable = getattr(self, '_reachable_floors', {self.floor})
            wc_floor = None
            if sim is not None:
                for fid in reachable:
                    if fid != self.floor and 'WC' in sim.shop.floors.get(fid, {}).get('walls', {}):
                        wc_floor = fid
                        break
            if wc_floor is not None:
                self.current_target_type = 'wc'
                self.target_floor = wc_floor
                self._route_to_floor(wc_floor, walls)
            else:
                # No reachable WC -- skip it
                self.visited_wc = True
                self.current_target_type = None

  


   def _checkout_lanes(self, walls):
        """All checkout lanes on this floor plan: the primary 'Checkout'
        plus any 'Checkout_LaneN' the architecture generator produced."""
        return [(nm, w) for nm, w in walls.items()
                if nm == 'Checkout' or nm.startswith('Checkout_Lane')]

   def _pick_checkout_lane(self, walls):
        """Least-loaded lane (fewest customers heading to / queuing at it),
        distance as the tie-breaker -- simple realistic queue balancing."""
        lanes = self._checkout_lanes(walls)
        if not lanes:
            return None
        loads = {nm: 0 for nm, _ in lanes}
        sim_ref = getattr(self, '_simulation_ref', None)
        if sim_ref is not None:
            try:
                for c in list(sim_ref.customers):
                    if c is self:
                        continue
                    ln = getattr(c, 'checkout_lane', None)
                    if ln in loads and (c.state == 'checking_out'
                                        or getattr(c, 'current_target_type', None) == 'checkout'):
                        loads[ln] += 1
            except Exception:
                pass

        def _key(entry):
            nm, w = entry
            px, py = w['position']
            sw, sh = w['size']
            d = math.hypot(px + sw / 2 - self.position[0],
                           py + sh / 2 - self.position[1])
            return (loads.get(nm, 0), d)

        return min(lanes, key=_key)

   def _set_checkout_target(self, walls):
        """Set target to a checkout lane. Checkout is always on floor 1.
        If not on floor 1, route there first. When the layout has several
        lanes, the customer queues at the least-loaded one."""
        self._checkout_arrived = False
        # Checkout lives on floor 1 -- route there first if needed
        if self.floor != 1:
            self.target_floor = 1
            self.current_target_type = 'checkout'
            self._route_to_floor(1, walls)
            return
        picked = self._pick_checkout_lane(walls)
        if picked is not None:
            nm, chk = picked
            self.checkout_lane = nm
            pos = chk['position']
            sz = chk['size']
            self.target_position = [
                pos[0] + sz[0]/2,
                pos[1] + sz[1]/2
            ]
            self._initialize_path(walls)
        else:
            self.dwell_time = np.random.uniform(5, 15)
            self.has_checked_out = True
            self.state = "exiting"
            self._set_exit_target(walls)
        



   def _set_exit_target(self, walls):
        """Set target to door. Door is always on floor 1.
        If not on floor 1, route there first."""
        self.current_target_type = 'exit'
        # Door lives on floor 1 -- route there first if needed
        if self.floor != 1:
            self.target_floor = 1
            self._route_to_floor(1, walls)
            return
        if self.door_position:
            self.target_position = list(self.door_position)
        else:
            self.target_position = self._find_exit_position()
        self._initialize_path(walls)


   def _handle_connector_arrival(self):
        """Customer has reached a connector tile. Verify they are actually
        on the source stair (not just within 0.5m through a wall), teleport
        them to the destination floor, then route to whatever they were
        originally trying to do (item, checkout, wc, exit, ...).

        Callable from any state that may end up routed through a connector:
        currently 'moving' (normal traversal) and 'exiting' (abandon-cart
        from a non-ground floor, or stuck-recovery that re-targets exit).
        On a successful teleport this method always changes state back to
        'moving' so the caller's state-specific logic restarts cleanly."""
        cid, dest_floor = self._pending_connector or (None, None)
        sim = getattr(self, '_simulation_ref', None)

        # Only fire the teleport if we're actually inside the source
        # stair tile. Without this guard a customer pushed against an
        # interior wall within 0.5 m of the stair center would visibly
        # teleport through the wall to the destination tile.
        on_stair = False
        if sim and cid is not None and cid in sim.shop.connectors:
            src_pair = sim.shop.connectors[cid].get(self.floor)
            src_walls = sim.shop.floors.get(self.floor, {}).get('walls', {})
            src_w = src_walls.get(src_pair) if src_pair else None
            if src_w is not None:
                sx, sy = src_w['position']
                sw, sh = src_w['size']
                margin = self.size * 0.5
                if (sx - margin <= self.position[0] <= sx + sw + margin
                        and sy - margin <= self.position[1] <= sy + sh + margin):
                    on_stair = True
        if not on_stair:
            # Not actually on the tile yet -- keep walking. _move_towards_target
            # was already called by the caller this tick.
            return

        if sim and cid is not None and cid in sim.shop.connectors:
            pair = sim.shop.connectors[cid].get(dest_floor)
            transitioned = False
            if pair is not None and dest_floor in sim.shop.floors:
                fwalls = sim.shop.floors[dest_floor]['walls']
                if pair in fwalls:
                    w = fwalls[pair]
                    self.position = [
                        w['position'][0] + w['size'][0] / 2,
                        w['position'][1] + w['size'][1] / 2,
                    ]
                    self.floor = dest_floor
                    sim.analytics.setdefault(
                        'floor_visits', defaultdict(int)
                    )[dest_floor] += 1
                    transitioned = True
            if transitioned:
                self._pending_connector = None

        if sim is not None:
            fwalls = sim.shop.floors.get(self.floor, {}).get('walls', {})
            all_items = sim.shop.all_items_across_floors()
            pending_type = getattr(self, '_pending_route_type', None)
            pending_item = getattr(self, '_pending_route_item', None)
            pending_dest = getattr(self, '_pending_route_dest', None)

            if pending_dest is not None and pending_dest != self.floor:
                self._route_to_floor(pending_dest, fwalls)
            elif pending_type == 'wc':
                self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                self.current_target_type = 'wc'
                self._set_wc_target(fwalls)
            elif pending_type in ('item', 'impulse') and pending_item in all_items:
                data = all_items[pending_item]
                if data.get('floor', self.floor) == self.floor:
                    self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                    self.current_target_type = pending_type
                    self.current_target_item = pending_item
                    pos, sz = data['position'], data['size']
                    self.target_position = [pos[0] + sz[0]/2, pos[1] + sz[1]/2]
                    self._initialize_path(fwalls)
                else:
                    self._route_to_floor(data.get('floor', self.floor), fwalls)
            elif pending_type == 'checkout':
                self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                self.current_target_type = 'checkout'
                self._set_checkout_target(fwalls)
            elif pending_type == 'exit':
                self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                self.current_target_type = 'exit'
                self._set_exit_target(fwalls)
            elif self.has_checked_out:
                self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                self._set_exit_target(fwalls)
            elif not self.shopping_list or all(
                i in self.visited_items for i in self.shopping_list
            ):
                self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                self.current_target_type = 'checkout'
                self._set_checkout_target(fwalls)
            else:
                self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                self._choose_next_target(all_items, fwalls)

        # Always end up in 'moving' so the standard target-arrival logic
        # (including LOS guards) handles the next leg.
        self._change_state("moving")


   def _reached_target(self):
           """Return True if close to target"""
           if self.target_position is None:
                return True
           dx = self.target_position[0] - self.position[0]
           dy = self.target_position[1] - self.position[1]
           return (dx*dx + dy*dy) < 0.25




   def _choose_next_target(self, shop_items, walls):
        """Pick next shopping target. Skips items on unreachable floors.
        `shop_items` is the flattened multi-floor catalogue (each entry
        has a 'floor' key)."""
        sim = getattr(self, '_simulation_ref', None)
        reachable = getattr(self, '_reachable_floors', None)
        if reachable is None and sim is not None:
            reachable = sim._reachable_floors(start_floor=1)
            self._reachable_floors = reachable
        if reachable is None:
            reachable = {self.floor}

        # Filter shopping list down to reachable items only.
        unvisited = [i for i in self.shopping_list
                    if i not in self.visited_items
                    and i in shop_items
                    and shop_items[i].get('floor', 1) in reachable]

        if not unvisited:
            # Nothing left to buy -> head to checkout (always floor 1).
            self.current_target_type = 'checkout'
            self.target_floor = 1
            if self.floor != 1:
                self._route_to_floor(1, walls)
            else:
                self._set_checkout_target(walls)
            return

        target_item = np.random.choice(unvisited)
        self.current_target_item = target_item
        self.current_target_type = 'item'
        data = shop_items[target_item]
        item_floor = data.get('floor', 1)
        self.target_floor = item_floor

        if item_floor != self.floor:
            self._route_to_floor(item_floor, walls)
        else:
            pos, sz = data['position'], data['size']
            self.target_position = [pos[0] + sz[0]/2, pos[1] + sz[1]/2]
            self._initialize_path(walls)


   def _route_to_floor(self, dest_floor, walls):
        """Walk to the nearest connector on this floor that links to dest_floor.
        If no such connector exists, drop the current target item from the
        shopping list and try to choose another reachable target instead of
        teleporting to the exit (which would look like a glitch)."""
        sim = getattr(self, '_simulation_ref', None)
        if sim is None:
            return

        if dest_floor == self.floor:
            # Already on the right floor -- nothing to do.
            return

        # Remember the original objective while walking through connector(s),
        # because current_target_type is temporarily set to 'connector'.
        if getattr(self, 'current_target_type', None) != 'connector':
            self._pending_route_type = getattr(self, 'current_target_type', None)
            self._pending_route_item = getattr(self, 'current_target_item', None)
            self._pending_route_dest = dest_floor

        connectors = getattr(sim.shop, 'connectors', {}) or {}
        candidates = []
        cur_floor_data = sim.shop.floors.get(self.floor, {})
        cur_floor_walls = cur_floor_data.get('walls', {})
        for cid, mapping in connectors.items():
            if self.floor in mapping and dest_floor in mapping:
                w = cur_floor_walls.get(mapping[self.floor])
                if w is not None:
                    candidates.append((cid, mapping[self.floor], w))

        if not candidates:
            # No direct connector on this floor -- try multi-hop via reachable
            # floors. If still impossible, drop the offending item and re-pick.
            reachable = sim._reachable_floors(start_floor=self.floor)
            if dest_floor not in reachable:
                # Truly unreachable from here. Forget this item.
                if getattr(self, 'current_target_item', None):
                    try:
                        item_name = self.current_target_item
                        self.shopping_list = [
                            i for i in self.shopping_list
                            if i != item_name
                        ]
                        # Mark as ATTEMPTED + ABANDONED, not visited; visited
                        # implies a purchase opportunity which we didn't have.
                        self.attempted_items.add(item_name)
                        self.abandoned_items.add(item_name)
                    except Exception:
                        pass
                # Fall back: head to checkout if on floor 1 else try to reach 1.
                if self.floor == 1:
                    self.current_target_type = 'checkout'
                    self._set_checkout_target(walls)
                else:
                    # Try to get back to floor 1 via any connector we can reach.
                    back_candidates = []
                    for cid, mapping in connectors.items():
                        if self.floor in mapping:
                            w2 = cur_floor_walls.get(mapping[self.floor])
                            if w2 is not None:
                                back_candidates.append((cid, mapping, w2))
                    if not back_candidates:
                        # Stuck on an island floor: instead of freezing in
                        # 'moving' with a null target, fall back to a local
                        # action so the customer keeps moving.
                        self.current_target_type = None
                        self.target_position = None
                        all_items_local = sim.shop.all_items_across_floors()
                        local = [n for n, d in all_items_local.items()
                                 if d.get('floor', self.floor) == self.floor
                                 and n not in self.attempted_items
                                 and n not in self.abandoned_items]
                        if local:
                            self._choose_next_target(all_items_local, walls)
                        else:
                            # No local item to visit -> head to exit
                            # (entrance on floor 1) and leave; never leave
                            # the customer with target=None in 'moving'.
                            self.current_target_type = 'exit'
                            self._set_exit_target(walls)
                        return
                    # Pick a connector that leads toward floor 1 when possible.
                    back_path = sim._connector_path(self.floor, 1) if hasattr(sim, '_connector_path') else []
                    desired_next = back_path[1] if len(back_path) >= 2 else None
                    cid, mapping, w2 = next(
                        ((cid0, map0, wall0) for cid0, map0, wall0 in back_candidates
                         if desired_next is None or desired_next in map0),
                        back_candidates[0]
                    )
                    # next_floor must differ from current floor; otherwise
                    # we attempt a self-loop teleport.
                    next_options = [f for f in sorted(mapping.keys()) if f != self.floor]
                    if desired_next is not None and desired_next != self.floor and desired_next in mapping:
                        next_floor = desired_next
                    elif next_options:
                        next_floor = next_options[0]
                    else:
                        # Connector only has self.floor in its mapping -> skip
                        # this connector entirely; fall back to local target.
                        self.current_target_type = None
                        self.target_position = None
                        all_items_local = sim.shop.all_items_across_floors()
                        local = [n for n, d in all_items_local.items()
                                 if d.get('floor', self.floor) == self.floor
                                 and n not in self.attempted_items
                                 and n not in self.abandoned_items]
                        if local:
                            self._choose_next_target(all_items_local, walls)
                        else:
                            self.current_target_type = 'exit'
                            self._set_exit_target(walls)
                        return
                    self._pending_connector = (cid, next_floor)
                    self.current_target_type = 'connector'
                    self.target_position = [w2['position'][0] + w2['size'][0]/2,
                                            w2['position'][1] + w2['size'][1]/2]
                    self._initialize_path(walls)
                return

            # dest_floor is reachable but not directly. Follow the shortest
            # connector path instead of hopping through an arbitrary floor.
            path = sim._connector_path(self.floor, dest_floor) if hasattr(sim, '_connector_path') else []
            next_floor = path[1] if len(path) >= 2 else None
            for cid, mapping in connectors.items():
                if self.floor in mapping and (next_floor is None or next_floor in mapping):
                    w2 = cur_floor_walls.get(mapping[self.floor])
                    if w2 is None:
                        continue
                    if next_floor is None:
                        next_floors = [f for f in mapping.keys() if f != self.floor]
                        if not next_floors:
                            continue
                        next_floor = next_floors[0]
                    self._pending_connector = (cid, next_floor)
                    self.current_target_type = 'connector'
                    self.target_position = [w2['position'][0] + w2['size'][0]/2,
                                            w2['position'][1] + w2['size'][1]/2]
                    self._initialize_path(walls)
                    return
            # Shouldn't get here, but be safe.
            return

        # Direct connector exists -- pick the nearest.
        def _dist(item):
            _, _, w = item
            cx = w['position'][0] + w['size'][0]/2
            cy = w['position'][1] + w['size'][1]/2
            return (cx - self.position[0])**2 + (cy - self.position[1])**2

        cid, name, w = min(candidates, key=_dist)
        self._pending_connector = (cid, dest_floor)
        self.current_target_type = 'connector'
        self.target_position = [w['position'][0] + w['size'][0]/2,
                                w['position'][1] + w['size'][1]/2]
        self._initialize_path(walls)



   def _move_towards_target(self, dt, walls):
        """Move towards target avoiding walls"""
        if self.target_position is None:
            return

        dx = self.target_position[0] - self.position[0]
        dy = self.target_position[1] - self.position[1]
        dist = math.hypot(dx, dy)
        if dist <= 0:
            return

        ux = dx/dist
        uy = dy/dist
        step = min(self.speed * dt, dist)

        # try full move
        nx = self.position[0] + ux * step
        ny = self.position[1] + uy * step
        if self._collides_with_interior(nx, ny, walls) == False:
            self.position[0] = nx
            self.position[1] = ny
        else:
            # try x only
            tx = self.position[0] + ux * step
            if self._collides_with_interior(tx, self.position[1], walls) == False:
                self.position[0] = tx
            else:
                # try y only
                ty = self.position[1] + uy * step
                if self._collides_with_interior(self.position[0], ty, walls) == False:
                    self.position[1] = ty
                else:
                    pass  # blocked

        # clamp to bounds
        w, h = self.shop_dimensions
        self.position[0] = max(0.1, min(w-0.1, self.position[0]))
        self.position[1] = max(0.1, min(h-0.1, self.position[1]))

    



   def _change_state(self, new_state):
        """Change state and track time"""
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            if new_state == 'checking_out':
                self.checkout_start_time = self.state_start_time



   def _find_exit_position(self):
        """Find exit pos"""
        if self.door_position:
            return list(self.door_position)
        else:
            w, h = self.shop_dimensions
            return [w/2, 0.1]



   def _collides_with_interior(self, x, y, walls):
    r = self.size
    sim = getattr(self, "_simulation_ref", None)
    if sim is not None:
        candidates = sim._candidate_walls(x, y, r, floor=self.floor)
        for wx, wy, ww, wh in candidates:
            if (wx - r <= x <= wx + ww + r) and (wy - r <= y <= wy + wh + r):
                return True
        # If the bin returned no candidates, do not short-circuit: walls may
        # exist that the spatial bin hasn't picked up yet (freshly rebuilt
        # geometry, missing floor entry, etc). Fall through to the explicit
        # walls scan so we never report 'no collision' on a non-empty floor.
        if candidates:
            return False

    for nm, data in walls.items():
        # Checkout lanes (all of them) and the WC are walk-up counters,
        # not solid obstacles -- same rule as the geometry/pathfinding caches.
        if nm.startswith('Section_') or nm.startswith('Checkout') or nm == 'WC':
            continue
        if data.get('category') == 'Connector':
            continue
        wx, wy = data['position']; ww, wh = data['size']
        if (wx - r <= x <= wx + ww + r) and (wy - r <= y <= wy + wh + r):
            return True
    return False
