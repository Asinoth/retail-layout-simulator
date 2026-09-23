import math
import numpy as np
from collections import defaultdict
import random  # not always used but sometimes handy


# TODO: maybe refactor this whole class later
from customer_pathfinding import (PathfindingMixin, AGENT_RADIUS_M,
                                  obstacle_rects)
from retail_literature import (
    IMPULSE_CHECKOUT_RADIUS_M, IMPULSE_DECAY_PER_ITEM,
    IMPULSE_BASE_PROPENSITY, IMPULSE_BROWSER_BONUS,
    IMPULSE_LONG_DWELL_BONUS, IMPULSE_LONG_DWELL_SECS,
    CHECKOUT_BASE_SECS, CHECKOUT_PER_ITEM_SECS,
    CHECKOUT_LOGNORM_SIGMA, CHECKOUT_MIN_SECS,
    BASKET_AFFINITY_PROB, BASKET_POP_SMOOTHING, LIST_LENGTH_BY_TYPE,
    STUCK_PROGRESS_M, STUCK_WINDOW_MOVING_S, STUCK_WINDOW_EXITING_S,
)
class Customer(PathfindingMixin):
   """Customer in shop sim. Handles movement, shopping, states etc."""


   def __init__(self, customer_id, start_pos, shop_items, shop_dimensions, door_position=None, door_side=None, floor=1, simulation_ref=None):
            self.id= customer_id
            # Set before the shopping list is drawn: the calibrated basket
            # draw reads the simulation's calibration through it.
            self._simulation_ref = simulation_ref
            # Simulated seconds seen by an agent driven without a simulation;
            # update() advances it by dt. See _now().
            self._local_clock = 0.0
            self.position = list(start_pos)
            self.target_position = None
            self.speed = np.random.uniform(0.8, 1.5)  #m/s
            self.shopping_list=[]
            self.visited_items = set()
            self.attempted_items = set()
            self.abandoned_items = set()
            # Stuck detection: anchor position plus simulated seconds
            # spent without moving STUCK_PROGRESS_M away from it.
            self._last_progress_pos = None
            self._no_progress_s = 0.0
            self.state = 'entering'
            # Every state the agent has been in, in order (no repeats).
            self.state_history = [self.state]
            self.spawn_sim_time = float(getattr(simulation_ref, 'sim_time', 0.0) or 0.0)
            self.dwell_time = 0
            self.start_shop_time = self._now()
            self.total_time_in_shop=0
            self.path = []
            self.path_index = 0
            self.shop_dimensions = shop_dimensions
            self.has_checked_out = False
            self._checkout_arrived = False
            self.checkout_lane = None   # which checkout lane this customer queues at
            self.checkout_wait_s = None # simulated seconds queued before service
            self._in_service = False
            self.abandon_probability = np.random.uniform(0.01, 0.05)
            self.abandon_cart = np.random.random() < self.abandon_probability
            self.entering_time = np.random.uniform(0.25,0.5)
            self.state_start_time = self._now()

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
            # True while a routing decision is being taken, so a fallback
            # that asks for another route is caught instead of recursing.
            self._routing = False
        
            # analytics tracking
            self.basket_value = 0.0
            self.zone_dwell_times = defaultdict(float)
            self.visited_zones = set()
            self.purchase_intent= np.random.uniform(0.7, 0.95)
            self.customer_type = np.random.choice(['quick', 'browser','thorough'], p=[0.3, 0.4, 0.3])
            self.loyalty_level = np.random.choice(["new", "regular", "vip"], p=[0.5, 0.35, 0.15])
            self.movement_path = []
        
            self._generate_shopping_list(shop_items)
            # Same radius the geometry caches inflate obstacles by, so a
            # free grid cell is a position this agent does not collide at.
            self.size = AGENT_RADIUS_M
    


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

   def _now(self):
        """Current simulated time in seconds.

        Every timestamp an agent takes comes from here, so state timing is
        on the same clock as arrivals and analytics and does not depend on
        how fast the host machine runs the ticks. An agent attached to a
        simulation reads its clock; one driven on its own counts the dt
        passed to update().
        """
        sim = getattr(self, '_simulation_ref', None)
        t = getattr(sim, 'sim_time', None) if sim is not None else None
        if t is None:
            return self._local_clock
        return float(t)

   def _shift_clock(self, offset):
        """Move every simulated-time stamp this agent holds by `offset`
        seconds, so elapsed times survive the simulation restarting its
        clock mid-visit."""
        self.state_start_time += offset
        self.start_shop_time += offset
        self.spawn_sim_time += offset
        for attr in ('checkout_start_time', 'checkout_end_time'):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, getattr(self, attr) + offset)
        self.movement_path[:] = [(x, y, t + offset)
                                 for x, y, t in self.movement_path]



   def _checkout_service_time(self):
        """Seconds this agent will occupy a checkout lane.

        Scales with the shopping-list items the agent actually reached and
        carries a lognormal right tail, rather than the flat draw this
        used to be. The basket dependence is the point: it closes the loop
        by which a layout that enlarges baskets also lengthens its own
        queues. Impulse pickups are not counted: they are chosen only
        after payment, so none exist yet when this is drawn and they add
        no service time.
        """
        n_items = len(self.visited_items)
        expected = CHECKOUT_BASE_SECS + CHECKOUT_PER_ITEM_SECS * n_items
        # Median multiplier 1.0, so `expected` is the median service time.
        secs = expected * float(np.random.lognormal(0.0, CHECKOUT_LOGNORM_SIGMA))
        return max(secs, CHECKOUT_MIN_SECS)

   def _calibration(self):
        """The simulation's calibration namespace, or an empty dict.

        This is the one way an agent reads dataset-derived inputs. Only
        ``analytics['calibration']`` is read: the same keys at top level
        are live counters filled from this simulator's own exits, and
        drawing baskets from them would let earlier agents' visits make
        already-popular items more popular.
        """
        sim_ref = getattr(self, '_simulation_ref', None)
        if sim_ref is None:
            return {}
        A = getattr(sim_ref, 'analytics', {}) or {}
        return A.get('calibration') or {}

   def _calibrated_basket_structure(self, regular_items):
        """(names, popularity weights, affinity map), or None.

        The transactional calibration keys both popularity and the
        co-purchase pairs by product description, which is what the
        layout builder uses for item names, so the join is direct; a
        product_id lookup is kept as a fallback for adapters that key by
        stock code instead. Returns None when no dataset has been loaded
        or nothing matches, leaving generated and synthetic shops on the
        original uniform draw.
        """
        sim_ref = getattr(self, '_simulation_ref', None)
        if sim_ref is None:
            return None

        cal = self._calibration()
        popular_src = cal.get('popular_items')
        pairs_src = cal.get('cross_merchandising')

        # Build once per assortment, not once per spawning agent: this
        # walks the whole item set and the pair table, and doing it on
        # every spawn starved the arrival loop badly enough to change
        # measured throughput. Seeding a calibration assigns fresh dicts,
        # so the cache holds the ones it was built from and compares by
        # identity; a re-seed over the same item names then rebuilds.
        cache_key = frozenset(regular_items)
        cached = getattr(sim_ref, '_basket_struct_cache', None)
        if (cached is not None and cached[0] == cache_key
                and cached[2] is popular_src and cached[3] is pairs_src):
            return cached[1]

        def _remember(value):
            sim_ref._basket_struct_cache = (cache_key, value,
                                            popular_src, pairs_src)
            return value

        popular = popular_src or {}
        pairs = pairs_src or {}
        if not popular and not pairs:
            return _remember(None)

        names = list(regular_items.keys())
        idx = {nm: i for i, nm in enumerate(names)}
        # Secondary route for adapters that key by stock code.
        alias = {}
        for nm, data in regular_items.items():
            pid = data.get('product_id')
            if pid is not None:
                alias[str(pid)] = nm

        def resolve(key):
            k = str(key)
            if k in idx:
                return k
            return alias.get(k)

        weights = np.zeros(len(names), dtype=float)
        matched = 0
        for key, cnt in popular.items():
            nm = resolve(key)
            if nm is not None:
                weights[idx[nm]] += float(cnt)
                matched += 1
        if matched == 0 and not pairs:
            return _remember(None)
        weights += BASKET_POP_SMOOTHING
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return _remember(None)
        weights = weights / total

        affinity = {}
        for key, cnt in pairs.items():
            if '|' not in str(key):
                continue
            a, b = str(key).split('|', 1)
            na, nb = resolve(a), resolve(b)
            if na is None or nb is None or na == nb:
                continue
            affinity.setdefault(na, []).append(nb)
            affinity.setdefault(nb, []).append(na)

        return _remember((names, idx, weights, affinity))

   def _draw_basket(self, regular_items, num):
        """Choose `num` distinct item names for this agent's list.

        With calibration present the draw is popularity-weighted and
        pulls co-purchase partners of items already chosen, so the
        assortment's real association structure reaches agent behaviour.
        Without it, this is the original uniform sample.
        """
        struct = self._calibrated_basket_structure(regular_items)
        if struct is None:
            return np.random.choice(list(regular_items.keys()),
                                    size=num, replace=False).tolist()

        names, idx, weights, affinity = struct
        chosen, chosen_set = [], set()

        # Sample without replacement by zeroing taken entries in a local
        # copy and drawing against the running cumulative sum. This is
        # hot -- it runs for every spawning agent -- so it avoids both
        # linear name lookups and np.random.choice's p= path.
        w = weights.copy()

        while len(chosen) < num:
            candidate = None
            # Extend an existing basket item along a co-purchase edge.
            if chosen and affinity and np.random.random() < BASKET_AFFINITY_PROB:
                seed_item = chosen[np.random.randint(len(chosen))]
                partners = [p for p in affinity.get(seed_item, ())
                            if p not in chosen_set]
                if partners:
                    candidate = partners[np.random.randint(len(partners))]
            if candidate is None:
                cum = np.cumsum(w)
                total = cum[-1]
                if total <= 0:
                    break
                candidate = names[int(np.searchsorted(
                    cum, np.random.random() * total))]
            chosen.append(candidate)
            chosen_set.add(candidate)
            w[idx[candidate]] = 0.0

        return chosen

   def _generate_shopping_list(self, shop_items):
        """Draw this agent's shopping list over the regular items (impulse
        displays, checkout and WC excluded).

        List LENGTH: when the calibration namespace holds a
        ``list_length_sample`` -- distinct stocked products per invoice,
        written by a transactional calibration seeded onto its own shop --
        the length is one value drawn uniformly from that sample, clipped
        to [1, number of regular items]. Otherwise it follows the
        type-conditional ``LIST_LENGTH_BY_TYPE`` law, an operational
        assumption kept for generated and synthetic shops. Either way it
        is one call on the global stream, in the place the type law's
        ``randint`` always had, so the spawn makes the same sequence of
        calls with or without a calibration (the basket draw that follows
        still depends on the length drawn).

        Under a calibration the length therefore no longer depends on
        customer type: the data give invoice sizes, not shopper types, and
        conditioning a measured size distribution on an unobserved label
        would reintroduce the assumption the sample replaces. Type still
        sets the impulse propensity (set below, plus the browser bonus at
        the checkout) and the washroom probability.

        List CONTENT is ``_draw_basket``'s: popularity-weighted with
        co-purchase pulls under a calibration, uniform without one.
        """
        # Per-type propensities are set before any early return: an agent
        # with an empty list still checks out, and the checkout and WC
        # logic read these every tick.
        cust_type = getattr(self, 'customer_type', "browser")
        self.impulse_probability = IMPULSE_BASE_PROPENSITY[cust_type]

        # wc prob
        self.wc_probability = {'quick': 0.15, 'browser': 0.35, 'thorough': 0.50}[cust_type]
        self.needs_wc = np.random.random() < self.wc_probability
        self.visited_wc = False

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
        
        sample = self._calibration().get('list_length_sample')
        if sample is not None and len(sample) > 0:
            drawn = int(sample[np.random.randint(len(sample))])
            num = min(max(drawn, 1), len(regular_items))
        else:
            min_items, max_items = LIST_LENGTH_BY_TYPE[cust_type]
            num = min(np.random.randint(min_items, max_items + 1),
                      len(regular_items))
        self.shopping_list = self._draw_basket(regular_items, num)
        
        # calc basket value
        val = 0.0
        for itm in self.shopping_list:
            val = val + shop_items[itm].get('price', 0.0)
        self.basket_value = val




    
   def _item_target_position(self, item_name, data):
        """Where to walk to shop `item_name`.

        Fixtures are solid, so the agent heads for the access point beside
        the shelving -- the free spot nearest its centre -- rather than the
        centre itself, which is inside the furniture. The simulation caches
        those with its geometry; without one the centre is the only thing
        available.
        """
        sim = getattr(self, '_simulation_ref', None)
        if sim is not None and hasattr(sim, 'item_access_point'):
            return sim.item_access_point(item_name, data,
                                         data.get('floor', self.floor))
        pos, size = data['position'], data['size']
        return [pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0]

   def _item_arrival_margin(self):
        """How far outside a fixture's rectangle counts as standing at it.

        The access point is a grid cell beside the inflated rectangle, so
        an agent that has arrived is up to its own radius plus one grid
        cell away from the fixture, plus a little slack for the step it
        stops on.
        """
        sim = getattr(self, '_simulation_ref', None)
        res = getattr(sim, 'path_grid_resolution', 0.25) if sim else 0.25
        return self.size + res + 0.05

   def _initialize_path(self, walls):
        """Plan waypoints around the walls and fixtures in the way"""
        self.path = self._compute_path(self.position,self.target_position, walls)
        self.path_index = 0
        # A new target starts a fresh no-progress window.
        self._reset_progress()
        if len(self.path) > 0:
            self.target_position = self.path[0]







   def update(self, dt, shop_items, walls, other_customers):
            """Main update loop - handles all states"""
            # dt is simulated seconds (the simulation has already scaled it
            # by simulation_speed), so every countdown and timestamp below
            # runs on simulated time.
            self._local_clock += dt
            current_time = self._now()
            self.total_time_in_shop += dt
            #print(f"updating customer {self.id}, state={self.state}")

            if self.state == "entering":
                time_in = current_time - self.state_start_time
                if time_in >= self.entering_time:
                    if len(self.shopping_list) == 0:
                        self._finish_shopping(walls)
                    else:
                        # State first, so a route that ends the visit
                        # (nothing reachable) keeps the state it sets.
                        self._change_state('moving')
                        self._choose_next_target(shop_items, walls)
                return

            elif self.state == "moving":
                if self.target_position is not None:
                    self._move_towards_target(dt, walls)

                    # If the agent has not got STUCK_PROGRESS_M away from
                    # its anchor within the moving window, it is stuck on a
                    # wall or has no valid path. Drop the target so it does
                    # not freeze visually.
                    if self._stalled(dt, STUCK_WINDOW_MOVING_S):
                        # Abandon current target and pick a new one.
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
                            self._finish_shopping(walls)
                        else:
                            self._choose_next_target(all_items, walls)
                        self._reset_progress()
                        return

                    if self._reached_target() == True:
                        if hasattr(self, 'path') and self.path_index + 1 < len(self.path):
                            self.path_index = self.path_index + 1
                            self.target_position = self.path[self.path_index]
                            return

                        ttype = getattr(self, 'current_target_type', None)

                        # Arrival guard for item/impulse/checkout/wc: only
                        # declare arrival once we are actually at the target
                        # -- beside the fixture, or at the counter -- and
                        # not merely within the waypoint tolerance of it,
                        # which can be on the far side of a wall.
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
                                margin = self._item_arrival_margin()
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
                            if self.has_checked_out:
                                # Back at the lane after paying (impulse
                                # pickups, or a WC trip routed via checkout).
                                # Re-entering checking_out would queue the
                                # agent at the lane a second time.
                                if self.needs_wc and self.visited_wc == False:
                                    self.current_target_type = 'wc'
                                    if self._set_wc_target(walls):
                                        return
                                self._change_state("exiting")
                                self._set_exit_target(walls)
                                return
                            self._change_state("checking_out")
                            self.dwell_time = self._checkout_service_time()
                            self._join_lane_queue()
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
                            # The zone roll-up counts 'WC' once per customer
                            # via visited_zones when the agent steps inside
                            # the rectangle; count here only if it has not,
                            # e.g. arrival just outside the edge.
                            if (getattr(self, '_simulation_ref', None) is not None
                                    and 'WC' not in self.visited_zones):
                                self.visited_zones.add('WC')
                                self._simulation_ref.analytics['area_visits']['WC'] += 1
                            return

                        # Connector arrival -- handle FIRST so we don't accidentally enter 'shopping'.
                        if ttype == 'connector':
                            self._handle_connector_arrival()
                            return

                        # Exit arrival from 'moving' state (stuck-recovery can
                        # set target_type='exit' without changing state). Hand
                        # off to 'exiting' so the simulation's step() can despawn
                        # the customer at the door.
                        if ttype == 'exit':
                            self._change_state('exiting')
                            return

                        # Default: arrived at an item shelf -> shop here.
                        self._change_state('shopping')
                        self.dwell_time = np.random.uniform(5, 15)
                        return
                elif self._stalled(dt, STUCK_WINDOW_MOVING_S):
                    # Nothing to walk to and nothing that will give this
                    # agent a target: it is as stuck as one pressed against
                    # a wall, so run the same window and then leave.
                    self._leave_through_door()
                return


            elif self.state == 'shopping':
                self.dwell_time = self.dwell_time - dt
                if not (self.dwell_time > 0):
                    # handle impulse queue
                    if hasattr(self,'impulse_queue'):
                        if len(self.impulse_queue) > 0:
                            # Same route as the first pickup: floor check
                            # plus a fresh A* path, not the previous leg's.
                            self._set_next_impulse_target(shop_items, walls)
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
                            self._finish_shopping(walls)
                        else:
                            self._change_state("moving")
                            self._choose_next_target(shop_items, walls)
                        return

                    if self.current_target_item:
                        self.visited_items.add(self.current_target_item)

                    # wc logic
                    if (self.needs_wc and self.visited_wc == False and
                        len(self.visited_items) >= len(self.shopping_list)//2):
                        self.current_target_type = 'wc'
                        if self._set_wc_target(walls):
                            self._change_state("moving")
                            return
                        # No reachable WC: carry on with checkout / next item.

                    if len(self.visited_items) >= len(self.shopping_list):
                        self._finish_shopping(walls)
                        return

                    self._change_state("moving")
                    self._choose_next_target(shop_items, walls)
                return

            elif self.state == 'checking_out':
                if self.has_checked_out == False:
                    # Each lane is a single FIFO server: only the agent at
                    # the head of its lane queue is served, the rest wait
                    # with their service time untouched.
                    if not self._in_service:
                        if not self._at_lane_head() or self._lane_freed_this_tick():
                            self.checkout_wait_s = (self.checkout_wait_s or 0.0) + dt
                            return
                        self._start_service()
                    self.dwell_time = self.dwell_time - dt
                    if not (self.dwell_time > 0):
                        self.has_checked_out = True
                        self.checkout_end_time = self._now()
                        self._leave_lane_queue()
                        self._attempt_impulse_purchases(shop_items, walls)
                        if getattr(self, 'impulse_queue', None):
                            return
                    return
          
                if self.needs_wc and self.visited_wc == False:
                    self.current_target_type = 'wc'
                    if self._set_wc_target(walls):
                        self._change_state("moving")
                        return
                self._change_state("exiting")
                self._set_exit_target(walls)
                return

            elif self.state == "exiting":
                if self.target_position:
                    end_of_path = False
                    if hasattr(self, 'path') and self._reached_target():
                        if self.path_index + 1 < len(self.path):
                            self.path_index = self.path_index + 1
                            self.target_position = self.path[self.path_index]
                        else:
                            end_of_path = True
                    self._move_towards_target(dt, walls)

                    # End of path -- if we were routed through a connector
                    # (e.g. abandon-cart from a non-ground floor), perform
                    # the floor teleport here too. Otherwise the customer
                    # would freeze at the stair center because the
                    # 'moving'-state connector handler never fires.
                    if (end_of_path
                            and getattr(self, 'current_target_type', None) == 'connector'):
                        self._handle_connector_arrival()
                        if self.state != 'exiting':
                            return

                    # Stuck-detection: if the agent has made no net progress
                    # toward the exit within the exiting window, hard-teleport
                    # to floor 1 + door so the simulation's step() can despawn
                    # it. Without this a customer walled off from the door
                    # (e.g. user moved walls after spawn) hangs on screen
                    # forever. Waypoint ticks count too, so the window is
                    # measured over the whole walk.
                    if self._stalled(dt, STUCK_WINDOW_EXITING_S):
                        sim = getattr(self, '_simulation_ref', None)
                        if sim is not None and sim.door_position is not None:
                            self.floor = 1
                            self.position = list(sim.door_position)
                            self.target_position = list(sim.door_position)
                            self.current_target_type = 'exit'
                            self._pending_connector = None
                            self._reset_progress()
                return




   def _attempt_impulse_purchases(self, shop_items, walls):
        """Check for impulse items near checkout"""
        if self.has_made_impulse_purchase == True:
            return

        impulse_candidates = []
        # Measure the catchment from the lane's centre. target_position is
        # by now the last A* waypoint, a grid-cell corner up to ~0.35 m off.
        key = getattr(self, 'checkout_lane', None) or 'Checkout'
        lane = walls.get(key) or walls.get('Checkout')
        if lane is not None:
            cx_co = lane['position'][0] + lane['size'][0] / 2
            cy_co = lane['position'][1] + lane['size'][1] / 2
        else:
            cx_co, cy_co = self.position

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
        self.target_position = self._item_target_position(next_item, data)
        self._change_state("moving")
        self._initialize_path(walls)




   def _set_wc_target(self, walls):
        """Set target to WC. If WC not on current floor, try to find it
        on another floor via connector.

        Returns False when no WC is reachable; the visit is then marked
        done and no target is set, so the caller must pick the next step
        itself rather than switch to 'moving' toward a stale target."""
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
            return True
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
                return True
            else:
                # No reachable WC -- skip it
                self.visited_wc = True
                self.current_target_type = None
                return False

  


   def _checkout_lanes(self, walls):
        """All checkout lanes on this floor plan: the primary 'Checkout'
        plus any 'Checkout_LaneN' the architecture generator produced."""
        return [(nm, w) for nm, w in walls.items()
                if nm == 'Checkout' or nm.startswith('Checkout_Lane')]

   def _pick_checkout_lane(self, walls):
        """Least-loaded lane (queue length plus unpaid customers already
        walking to it), distance as the tie-breaker -- simple realistic
        queue balancing."""
        lanes = self._checkout_lanes(walls)
        if not lanes:
            return None
        loads = {nm: 0 for nm, _ in lanes}
        sim_ref = getattr(self, '_simulation_ref', None)
        if sim_ref is not None:
            try:
                queues = getattr(sim_ref, 'lane_queues', None) or {}
                for nm in loads:
                    loads[nm] += len(queues.get(nm, ()))
                # Agents at the lane are already in its queue; add the ones
                # still on their way. Paid agents walking back past the
                # lane will not queue again.
                for c in list(sim_ref.customers):
                    if c is self:
                        continue
                    ln = getattr(c, 'checkout_lane', None)
                    if (ln in loads and c.state != 'checking_out'
                            and not c.has_checked_out
                            and getattr(c, 'current_target_type', None) == 'checkout'):
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

   def _lane_queue(self):
        """The FIFO for this agent's lane on the simulation (head is in
        service), or None when there is no simulation to queue on."""
        sim = getattr(self, '_simulation_ref', None)
        queues = getattr(sim, 'lane_queues', None) if sim is not None else None
        if queues is None:
            return None
        return queues.setdefault(self.checkout_lane or 'Checkout', [])

   def _join_lane_queue(self):
        """Take a place at the back of the lane queue on arrival."""
        self.checkout_wait_s = 0.0
        self._in_service = False
        q = self._lane_queue()
        if q is not None and self.id not in q:
            q.append(self.id)

   def _at_lane_head(self):
        """True when this agent is the one the lane serves next."""
        q = self._lane_queue()
        if q is None:
            return True
        if self.id not in q:
            # The queues were cleared under us; rejoin at the back rather
            # than wait for a turn that can never come.
            q.append(self.id)
        return q[0] == self.id

   def _start_service(self):
        """Begin service at the head of the queue and log the wait."""
        self._in_service = True
        self.checkout_start_time = self._now()
        wait = self.checkout_wait_s or 0.0
        self.checkout_wait_s = wait
        sim = getattr(self, '_simulation_ref', None)
        if sim is not None:
            sim.analytics.setdefault('queue_wait_times', []).append(wait)

   def _lane_freed_this_tick(self):
        """True when the customer ahead finished during the current tick.

        Agents update one after another within a tick, so without this a
        successor that happens to update later would start service in a
        tick the lane already spent on the previous customer. The lane's
        release stamp is consumed either way.
        """
        sim = getattr(self, '_simulation_ref', None)
        stamps = getattr(sim, 'lane_released_at', None) if sim is not None else None
        if not stamps:
            return False
        released = stamps.pop(self.checkout_lane or 'Checkout', None)
        return released is not None and released == getattr(sim, 'sim_time', None)

   def _leave_lane_queue(self):
        """Free the lane for the next agent once service completes."""
        q = self._lane_queue()
        if q is not None and self.id in q:
            q.remove(self.id)
            sim = self._simulation_ref
            stamps = getattr(sim, 'lane_released_at', None)
            if stamps is not None:
                stamps[self.checkout_lane or 'Checkout'] = getattr(sim, 'sim_time', None)

   def _finish_shopping(self, walls):
        """End the shopping phase: go to checkout, or straight for the door
        when the agent abandons its cart.

        Every route that would send an unpaid agent to checkout comes
        through here, so the abandonment draw is honoured whichever way
        the visit ended (list done, WC trip, stuck target, unreachable
        floor). The state is set here too, so callers must not override it.
        """
        if self.abandon_cart and not self.has_checked_out:
            self._change_state('exiting')
            self._set_exit_target(walls)
            return
        self.current_target_type = 'checkout'
        self._change_state('moving')
        self._set_checkout_target(walls)

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
            self._change_state("exiting")
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


   def _leave_through_door(self):
        """Last resort for an agent that cannot route anywhere.

        A floor whose connectors have all been removed leaves nothing to
        satisfy -- not an item, not the checkout, not the exit -- because
        every one of those routes goes through a connector. Rather than
        ask routing again (which would ask routing again), put the agent
        at the entrance on floor 1 and let step() despawn it, the same
        way the exiting stall detector does.
        """
        sim = getattr(self, '_simulation_ref', None)
        self._pending_connector = None
        self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
        self.current_target_type = 'exit'
        self.path = []
        self.path_index = 0
        if sim is not None and sim.door_position is not None:
            self.floor = 1
            self.position = list(sim.door_position)
            self.target_position = list(sim.door_position)
        self._change_state('exiting')
        self._reset_progress()

   def _handle_connector_arrival(self):
        """Customer has reached a connector tile. Verify they are actually
        on the source stair (not just within 0.5m through a wall), teleport
        them to the destination floor, then route to whatever they were
        originally trying to do (item, checkout, wc, exit, ...).

        Callable from any state that may end up routed through a connector:
        currently 'moving' (normal traversal) and 'exiting' (abandon-cart
        from a non-ground floor, or stuck-recovery that re-targets exit).
        On a successful teleport this method changes state back to 'moving'
        so the caller's state-specific logic restarts cleanly ('exiting' when
        the next leg ends the visit)."""
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

        # Always end up in 'moving' so the standard target-arrival logic
        # (including LOS guards) handles the next leg. Set it before the
        # re-routing below, so a route that ends the visit (abandoned cart,
        # no checkout lane) keeps the 'exiting' state it chose.
        self._change_state("moving")

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
                if not self._set_wc_target(fwalls):
                    # WC no longer reachable (layout edited mid-trip): carry
                    # on with the visit instead of idling on the stair tile.
                    if self.has_checked_out:
                        self._set_exit_target(fwalls)
                    else:
                        self._choose_next_target(all_items, fwalls)
            elif pending_type in ('item', 'impulse') and pending_item in all_items:
                data = all_items[pending_item]
                if data.get('floor', self.floor) == self.floor:
                    self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                    self.current_target_type = pending_type
                    self.current_target_item = pending_item
                    self.target_position = self._item_target_position(
                        pending_item, data)
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
                self._finish_shopping(fwalls)
            else:
                self._pending_route_type = self._pending_route_item = self._pending_route_dest = None
                self._choose_next_target(all_items, fwalls)


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
            # Nothing left to buy -> checkout (always floor 1; the checkout
            # target routes there), or the door if the cart is abandoned.
            self._finish_shopping(walls)
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
            self.target_position = self._item_target_position(target_item, data)
            self._initialize_path(walls)


   def _unvisited_on_this_floor(self, all_items):
        """Shopping-list items still to buy that sit on the agent's floor.

        The fallback for a stranded agent only helps if it names something
        the agent would actually walk to: any other item on the floor is
        not on its list and choosing a target would look past it.
        """
        return [n for n in self.shopping_list
                if n in all_items
                and all_items[n].get('floor', self.floor) == self.floor
                and n not in self.visited_items
                and n not in self.attempted_items
                and n not in self.abandoned_items]

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

        if getattr(self, '_routing', False):
            # Routing is already running: the fallback it picked asked for
            # another route, so this floor cannot serve that objective
            # either. Leave instead of recursing between the two.
            self._leave_through_door()
            return
        self._routing = True
        try:
            self._route_to_floor_inner(dest_floor, walls, sim)
        finally:
            self._routing = False

   def _route_to_floor_inner(self, dest_floor, walls, sim):
        """Body of _route_to_floor, which holds the re-entrancy guard."""
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
                    self._finish_shopping(walls)
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
                        local = self._unvisited_on_this_floor(all_items_local)
                        if local:
                            self._choose_next_target(all_items_local, walls)
                        else:
                            # Nothing left to buy here, and checkout and the
                            # door are both off this floor, so there is no
                            # route to walk: leave from the entrance.
                            self._leave_through_door()
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
                        local = self._unvisited_on_this_floor(all_items_local)
                        if local:
                            self._choose_next_target(all_items_local, walls)
                        else:
                            self._leave_through_door()
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
        """Move towards target, refusing steps into walls and fixtures"""
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
            # Slide along one axis instead: a full step towards the target's
            # coordinate on that axis, landing on it exactly when it is
            # closer than a step. Aisle waypoints can lie right on the edge
            # of a fixture's inflated footprint, and an agent that rounds a
            # shelf corner early has to cover that last sliver sideways
            # before it can walk on. A slide scaled by the direction cosine
            # shrinks as the sliver does, so crossing a thin one takes longer
            # than the stall window and the agent gives up on its target.
            step_len = self.speed * dt
            tx = (self.target_position[0] if abs(dx) <= step_len
                  else self.position[0] + math.copysign(step_len, dx))
            ty = (self.target_position[1] if abs(dy) <= step_len
                  else self.position[1] + math.copysign(step_len, dy))
            # try x only
            if dx != 0 and self._collides_with_interior(tx, self.position[1], walls) == False:
                self.position[0] = tx
            # try y only
            elif dy != 0 and self._collides_with_interior(self.position[0], ty, walls) == False:
                self.position[1] = ty
            else:
                pass  # blocked

        # clamp to bounds
        w, h = self.shop_dimensions
        self.position[0] = max(0.1, min(w-0.1, self.position[0]))
        self.position[1] = max(0.1, min(h-0.1, self.position[1]))

    



   def _change_state(self, new_state):
        """Change state, track time and log the state in state_history.
        All state assignments after construction go through here."""
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = self._now()
            self.state_history.append(new_state)
            # Progress (or its absence) in the previous state says nothing
            # about the new one, so the stuck window starts over.
            self._reset_progress()

   def _reset_progress(self):
        """Re-anchor the stuck detector at the current position."""
        self._last_progress_pos = (self.position[0], self.position[1])
        self._no_progress_s = 0.0

   def _stalled(self, dt, window_s):
        """Advance the no-progress clock by `dt` simulated seconds and
        report whether it has run past `window_s`.

        Progress is net displacement from the anchor, not the length of a
        single tick's step: a normal step (speed * dt) is only a few
        centimetres, so a per-tick test flags ordinary walking as stuck.
        """
        if self._last_progress_pos is None:
            self._reset_progress()
            return False
        dx = self.position[0] - self._last_progress_pos[0]
        dy = self.position[1] - self._last_progress_pos[1]
        if math.hypot(dx, dy) >= STUCK_PROGRESS_M:
            self._reset_progress()
            return False
        self._no_progress_s += dt
        return self._no_progress_s > window_s



   def _find_exit_position(self):
        """Find exit pos"""
        if self.door_position:
            return list(self.door_position)
        else:
            w, h = self.shop_dimensions
            return [w/2, 0.1]



   def _collides_with_interior(self, x, y, walls):
    r = self.size
    items = None
    sim = getattr(self, "_simulation_ref", None)
    if sim is not None:
        candidates = sim._candidate_walls(x, y, r, floor=self.floor)
        for wx, wy, ww, wh in candidates:
            if (wx - r <= x <= wx + ww + r) and (wy - r <= y <= wy + wh + r):
                return True
        # Once this floor has bins the candidate query is exhaustive for
        # radius r (it covers neighbouring bins, or scans the floor's
        # obstacles when its bins are empty), so an empty result just means
        # nothing solid nearby. A floor with no bins entry, or bins awaiting
        # a rebuild after a layout edit, still falls through to the explicit
        # scan.
        bins_by_floor = getattr(sim, 'wall_bins_by_floor', None) or {}
        if candidates or (self.floor in bins_by_floor
                          and not getattr(sim, 'geometry_dirty', False)):
            return False
        floors = getattr(getattr(sim, 'shop', None), 'floors', None) or {}
        items = (floors.get(self.floor) or {}).get('items')

    # Fixtures are obstacles as well as walls. An agent driven without a
    # simulation has no catalogue to read them from here and collides with
    # the walls it was handed.
    for wx, wy, ww, wh in obstacle_rects(walls, items):
        if (wx - r <= x <= wx + ww + r) and (wy - r <= y <= wy + wh + r):
            return True
    return False
