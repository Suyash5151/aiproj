import heapq
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

class DeliveryMode(Enum):
    NON_INCREMENTAL = 'N'
    INCREMENTAL = 'I'

class ResourceType(Enum):
    DOCUMENT = 'document'
    STYLESHEET = 'stylesheet'
    SCRIPT = 'script'
    FONT = 'font'
    IMAGE = 'image'
    MEDIA = 'media'
    XHR = 'xhr'
    FETCH = 'fetch'

@dataclass
class NetworkCondition:
    """Models dynamic network conditions"""
    base_bandwidth: float = 500.0  # bytes per time unit
    latency: float = 50.0  # milliseconds
    jitter: float = 10.0  # variance in latency
    packet_loss: float = 0.01  # 1% packet loss
    
    def get_effective_bandwidth(self, time: float) -> float:
        """Simulate bandwidth fluctuation over time"""
        variance = 0.2 * self.base_bandwidth * (random.random() - 0.5)
        return max(100, self.base_bandwidth + variance)
    
    def get_latency(self) -> float:
        """Return latency with jitter"""
        return self.latency + random.uniform(-self.jitter, self.jitter)

@dataclass(order=True)
class Request:
    """Enhanced request with dependencies and priority boost"""
    urgency: int = field(compare=True)
    arrival_time: float = field(compare=True)
    id: int = field(compare=False)
    type: ResourceType = field(compare=False)
    chrom_prio: int = field(compare=False)
    size: int = field(compare=False)
    mode: DeliveryMode = field(compare=False)
    dependencies: List[int] = field(default_factory=list, compare=False)
    priority_boost: int = field(default=0, compare=False)
    start_time: Optional[float] = field(default=None, compare=False)
    finish_time: Optional[float] = field(default=None, compare=False)
    bytes_sent: int = field(default=0, compare=False)
    blocked: bool = field(default=False, compare=False)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    
    def effective_urgency(self) -> int:
        """Calculate urgency with priority boost"""
        return max(0, self.urgency - self.priority_boost)
    
    def is_complete(self) -> bool:
        return self.bytes_sent >= self.size
    
    def remaining_bytes(self) -> int:
        return self.size - self.bytes_sent

class EPSScheduler:
    """Enhanced EPS scheduler with dynamic prioritization and congestion control"""
    
    # Extended mapping table
    MAPPING = {
        (ResourceType.DOCUMENT, 0): (0, DeliveryMode.NON_INCREMENTAL),
        (ResourceType.STYLESHEET, 0): (1, DeliveryMode.NON_INCREMENTAL),
        (ResourceType.FONT, 0): (1, DeliveryMode.NON_INCREMENTAL),
        (ResourceType.SCRIPT, 0): (1, DeliveryMode.NON_INCREMENTAL),
        (ResourceType.DOCUMENT, 1): (2, DeliveryMode.NON_INCREMENTAL),
        (ResourceType.STYLESHEET, 1): (2, DeliveryMode.NON_INCREMENTAL),
        (ResourceType.SCRIPT, 1): (2, DeliveryMode.NON_INCREMENTAL),
        (ResourceType.FONT, 1): (3, DeliveryMode.INCREMENTAL),
        (ResourceType.DOCUMENT, 2): (3, DeliveryMode.NON_INCREMENTAL),
        (ResourceType.STYLESHEET, 2): (4, DeliveryMode.INCREMENTAL),
        (ResourceType.SCRIPT, 2): (4, DeliveryMode.INCREMENTAL),
        (ResourceType.DOCUMENT, 3): (4, DeliveryMode.INCREMENTAL),
        (ResourceType.STYLESHEET, 3): (5, DeliveryMode.INCREMENTAL),
        (ResourceType.FONT, 3): (5, DeliveryMode.INCREMENTAL),
        (ResourceType.IMAGE, 3): (5, DeliveryMode.INCREMENTAL),
        (ResourceType.DOCUMENT, 4): (6, DeliveryMode.INCREMENTAL),
        (ResourceType.IMAGE, 4): (7, DeliveryMode.INCREMENTAL),
        (ResourceType.MEDIA, 4): (7, DeliveryMode.INCREMENTAL),
    }
    
    DIRECT_MAP = {0: 0, 1: 2, 2: 3, 3: 5, 4: 7}
    
    def __init__(self, network: NetworkCondition, quantum: float = 100.0):
        self.network = network
        self.quantum = quantum  # bytes per scheduling quantum
        self.non_inc_queue: List[Request] = []
        self.inc_queue: deque = deque()
        self.blocked_queue: List[Request] = []
        self.completed: List[Request] = []
        self.time = 0.0
        self.total_bytes_sent = 0
        self.congestion_window = 10  # Number of concurrent incremental streams
        
    def classify_request(self, req: Request) -> Tuple[int, DeliveryMode]:
        """Determine EPS urgency and delivery mode"""
        key = (req.type, req.chrom_prio)
        if key in self.MAPPING:
            urgency, mode = self.MAPPING[key]
        else:
            urgency = self.DIRECT_MAP.get(req.chrom_prio, 7)
            mode = DeliveryMode.INCREMENTAL
        return urgency, mode
    
    def add_request(self, req: Request):
        """Add request to appropriate queue"""
        req.urgency, req.mode = self.classify_request(req)
        
        # Check if blocked by dependencies
        if req.dependencies:
            req.blocked = True
            self.blocked_queue.append(req)
        elif req.mode == DeliveryMode.NON_INCREMENTAL:
            heapq.heappush(self.non_inc_queue, req)
        else:
            self.inc_queue.append(req)
    
    def unblock_requests(self, completed_id: int):
        """Unblock requests that depended on completed request"""
        newly_unblocked = []
        remaining = []
        
        for req in self.blocked_queue:
            if completed_id in req.dependencies:
                req.dependencies.remove(completed_id)
            
            if not req.dependencies:
                req.blocked = False
                newly_unblocked.append(req)
            else:
                remaining.append(req)
        
        self.blocked_queue = remaining
        
        # Add unblocked requests to appropriate queues
        for req in newly_unblocked:
            if req.mode == DeliveryMode.NON_INCREMENTAL:
                heapq.heappush(self.non_inc_queue, req)
            else:
                self.inc_queue.append(req)
    
    def serve_non_incremental(self):
        """Serve non-incremental queue with priority"""
        while self.non_inc_queue:
            req = heapq.heappop(self.non_inc_queue)
            
            # Simulate packet loss and retries
            if random.random() < self.network.packet_loss and req.retries < req.max_retries:
                req.retries += 1
                self.time += self.network.get_latency() / 1000.0
                heapq.heappush(self.non_inc_queue, req)
                continue
            
            # Add initial latency
            if req.start_time is None:
                req.start_time = self.time + self.network.get_latency() / 1000.0
                self.time = req.start_time
            
            # Send entire request
            bandwidth = self.network.get_effective_bandwidth(self.time)
            send_time = req.size / bandwidth
            self.time += send_time
            req.bytes_sent = req.size
            req.finish_time = self.time
            self.total_bytes_sent += req.size
            self.completed.append(req)
            
            # Unblock dependent requests
            self.unblock_requests(req.id)
    
    def serve_incremental(self):
        """Serve incremental queue with weighted fair sharing"""
        if not self.inc_queue:
            return
        
        # Limit concurrent streams (congestion control)
        active_reqs = list(self.inc_queue)[:self.congestion_window]
        
        # Calculate weights based on effective urgency
        weights = [8 - req.effective_urgency() for req in active_reqs]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return
        
        # Round-robin scheduling with weighted quantums
        all_complete = False
        while not all_complete:
            all_complete = True
            bandwidth = self.network.get_effective_bandwidth(self.time)
            
            for req, weight in zip(active_reqs, weights):
                if req.is_complete():
                    continue
                
                all_complete = False
                
                # Add initial latency for first send
                if req.start_time is None:
                    req.start_time = self.time + self.network.get_latency() / 1000.0
                    self.time = req.start_time
                
                # Calculate quantum based on weight
                share = weight / total_weight
                quantum_bytes = min(self.quantum * share, req.remaining_bytes())
                
                # Simulate packet loss
                if random.random() < self.network.packet_loss and req.retries < req.max_retries:
                    req.retries += 1
                    self.time += self.network.get_latency() / 1000.0
                    continue
                
                # Send quantum
                send_time = quantum_bytes / bandwidth
                self.time += send_time
                req.bytes_sent += quantum_bytes
                self.total_bytes_sent += quantum_bytes
                
                # Check if complete
                if req.is_complete():
                    req.finish_time = self.time
                    self.completed.append(req)
                    self.unblock_requests(req.id)
        
        # Clear completed requests from queue
        self.inc_queue = deque([r for r in self.inc_queue if not r.is_complete()])
    
    def run(self):
        """Execute the scheduling simulation"""
        self.serve_non_incremental()
        self.serve_incremental()
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        """Calculate performance statistics"""
        if not self.completed:
            return {}
        
        finish_times = [r.finish_time for r in self.completed]
        latencies = [r.finish_time - r.arrival_time for r in self.completed]
        
        return {
            'total_requests': len(self.completed),
            'total_time': max(finish_times),
            'total_bytes': self.total_bytes_sent,
            'avg_throughput': self.total_bytes_sent / max(finish_times),
            'avg_latency': sum(latencies) / len(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'total_retries': sum(r.retries for r in self.completed)
        }

def generate_realistic_requests(count: int = 20) -> List[Request]:
    """Generate realistic web resource requests with dependencies"""
    requests = []
    
    # Always start with HTML document
    requests.append(Request(
        id=1,
        type=ResourceType.DOCUMENT,
        chrom_prio=0,
        size=random.randint(5000, 15000),
        urgency=0,
        mode=DeliveryMode.NON_INCREMENTAL,
        arrival_time=0.0
    ))
    
    # Critical CSS and fonts (depend on HTML)
    for i in range(2, min(5, count)):
        requests.append(Request(
            id=i,
            type=random.choice([ResourceType.STYLESHEET, ResourceType.FONT]),
            chrom_prio=random.randint(0, 1),
            size=random.randint(500, 3000),
            urgency=0,
            mode=DeliveryMode.NON_INCREMENTAL,
            dependencies=[1],
            arrival_time=random.uniform(0.01, 0.05)
        ))
    
    # Scripts (some depend on CSS)
    for i in range(5, min(10, count)):
        requests.append(Request(
            id=i,
            type=ResourceType.SCRIPT,
            chrom_prio=random.randint(0, 2),
            size=random.randint(1000, 8000),
            urgency=0,
            mode=DeliveryMode.NON_INCREMENTAL,
            dependencies=[random.randint(1, i-1)] if random.random() > 0.5 else [],
            arrival_time=random.uniform(0.05, 0.15)
        ))
    
    # Images and media (lower priority)
    for i in range(10, count + 1):
        requests.append(Request(
            id=i,
            type=random.choice([ResourceType.IMAGE, ResourceType.MEDIA]),
            chrom_prio=random.randint(3, 4),
            size=random.randint(2000, 20000),
            urgency=0,
            mode=DeliveryMode.INCREMENTAL,
            arrival_time=random.uniform(0.1, 0.3)
        ))
    
    return requests

# Main simulation
if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Create network condition (simulate 4G connection)
    network = NetworkCondition(
        base_bandwidth=500.0,
        latency=50.0,
        jitter=15.0,
        packet_loss=0.02
    )
    
    # Generate realistic requests
    requests = generate_realistic_requests(25)
    
    # Create scheduler and add requests
    scheduler = EPSScheduler(network, quantum=150.0)
    for req in requests:
        scheduler.add_request(req)
    
    # Run simulation
    stats = scheduler.run()
    
    # Display results
    print("=" * 100)
    print("EPS SCHEDULING SIMULATION RESULTS")
    print("=" * 100)
    print(f"\nNetwork Conditions:")
    print(f"  Base Bandwidth: {network.base_bandwidth:.1f} bytes/time unit")
    print(f"  Latency: {network.latency:.1f}ms (±{network.jitter:.1f}ms jitter)")
    print(f"  Packet Loss: {network.packet_loss * 100:.1f}%")
    
    print(f"\nPerformance Statistics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Total Time: {stats['total_time']:.2f} time units")
    print(f"  Total Data: {stats['total_bytes']} bytes")
    print(f"  Avg Throughput: {stats['avg_throughput']:.2f} bytes/time unit")
    print(f"  Avg Latency: {stats['avg_latency']:.2f} time units")
    print(f"  Latency Range: {stats['min_latency']:.2f} - {stats['max_latency']:.2f}")
    print(f"  Total Retries: {stats['total_retries']}")
    
    print(f"\n{'ID':<4} {'Type':<12} {'Prio':<5} {'Urg':<4} {'Mode':<5} {'Size':<7} {'Start':<8} {'Finish':<8} {'Deps':<8} {'Retries':<8}")
    print("-" * 100)
    
    for req in sorted(scheduler.completed, key=lambda r: r.finish_time):
        deps_str = ','.join(map(str, req.dependencies)) if req.dependencies else '-'
        print(f"{req.id:<4} {req.type.value:<12} {req.chrom_prio:<5} {req.urgency:<4} "
              f"{req.mode.value:<5} {req.size:<7} {req.start_time:<8.2f} {req.finish_time:<8.2f} "
              f"{deps_str:<8} {req.retries:<8}")
    
    print("\n" + "=" * 100)