
```mermaid
flowchart TD
    A[User App Flutter/React] -->|API Calls| B[Backend API Django/Node.js]
    B --> C[AI Agent - Trip Generator]
    B --> D[Database PostgreSQL/MongoDB]
    B --> E[Payment Gateway Stripe/Razorpay]
    B --> F[Broadcast Service - WebSocket/GeoAPI]
    D -->|Stores| G[(Data Entities: User, Trip, Rider, Community, Payment)]
    F --> H[Nearby Users]
```


---

## 🧠 Entity Relationship Diagram (ERD)

```mermaid
erDiagram
    USER ||--o{ TRIP : creates
    USER ||--o{ BROADCAST : sends
    USER ||--o{ FEEDBACK : gives
    USER ||--o{ PAYMENT : makes
    USER ||--o{ COMMUNITY : joins

    RIDER ||--o{ TRIP : assigned
    TRIP ||--|{ COMMUNITY : associated
    TRIP ||--|{ PAYMENT : includes

    USER {
        string uname PK
        string email
        string password
        float lat
        float lon
        string usertype
        boolean isBroadcast
    }

    RIDER {
        string uname FK
        float rating
        boolean isVerified
        boolean isActive
        string location
    }

    TRIP {
        string tname
        date start_date
        date end_date
        string riderid
        string userid
        string location
        float cost
        string planning
        string OTP
    }

    COMMUNITY {
        string name
        string tripid FK
        list members
        list msg
    }

    BROADCAST {
        int id
        string user FK
        list msg
        float fee
    }

    FEEDBACK {
        int id
        string user FK
        string rider FK
        int rating
        string comment
        enum type
    }

    PAYMENT {
        int id
        string userid FK
        string tripid FK
        float amount
        string status
        string transaction_id
    }
```

---

## 🧩 Use Case Diagram

```mermaid
graph TD
    User -->|Request Trip| AI_Agent
    AI_Agent -->|Generate Plan| Trip
    User -->|Make Payment| Payment
    Rider -->|Accept/Reject Trip| Trip
    Trip -->|Add Members| Community
    Community -->|Send Messages| Members
    User -->|Broadcast Message| Broadcast
    Broadcast -->|Notify Nearby Users| User
    User -->|Give Feedback| Feedback
```

---

## ⏱️ Sequence Diagram – Trip Booking Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as AI Agent
    participant B as Backend
    participant R as Rider
    participant P as Payment Gateway

    U->>B: Send trip preferences
    B->>A: Generate AI trip plan
    A-->>B: Return plan metadata
    B-->>U: Display AI-generated plan
    U->>B: Confirm booking
    B->>R: Send ride request
    R-->>B: Accept trip
    B->>P: Process payment
    P-->>B: Payment confirmed
    B-->>U: Trip confirmed
    B-->>R: Trip details + OTP
```

---

## ⚙️ Level 0 DFD (Data Flow Diagram)

```mermaid
flowchart TD
    User -->|Trip Request| Process1[AI Trip Planner]
    Process1 -->|Trip Plan| User
    User -->|Ride Request| Process2[Rider Matching]
    Process2 -->|Ride Acceptance| Rider
    User -->|Payment Details| Process3[Payment System]
    Process3 -->|Confirmation| User
```

---

## ⚙️ Level 1 DFD (Expanded)

```mermaid
flowchart TD
    subgraph AI_Trip_Planner
        A1[Collect Preferences] --> A2[Generate Itinerary]
        A2 --> A3[Store Trip Metadata]
    end

    subgraph Rider_Matching
        R1[Fetch Active Riders] --> R2[Send Trip Requests]
        R2 --> R3[Verify Bids via AI]
    end

    subgraph Payment_System
        P1[Initiate Transaction] --> P2[Verify Payment]
        P2 --> P3[Update Trip Status]
    end

    User --> A1
    A3 --> R1
    R3 --> P1
```

---

## 🧱 Deployment Diagram

```mermaid
graph TD
    A[Mobile/Web Client] --> B[API Gateway]
    B --> C[Backend Server]
    C --> D[(Database)]
    C --> E[AI Agent Server]
    C --> F[Payment Gateway]
    C --> G[Broadcast & Notification Server]
```



