% MiniZinc Model: Recursive CRUD Review Optimization
% Find optimal pricing and expert allocation to maximize revenue

% ============================================================================
% PARAMETERS (Input)
% ============================================================================

% Number of ZK experts available
int: num_experts;

% Number of CRUD cells (CREATE, READ, UPDATE, DELETE)
int: num_crud_cells = 4;

% Maximum experts per review
int: max_experts_per_review;

% Cost per expert per review (base rate)
int: base_expert_cost;

% ============================================================================
% DECISION VARIABLES (What MiniZinc finds)
% ============================================================================

% Price per review level (to be optimized)
array[1..max_experts_per_review] of var int: price_per_level;

% Number of customers at each tier (to be optimized)
array[1..max_experts_per_review] of var int: customers_per_tier;

% Expert allocation per tier
array[1..max_experts_per_review] of var int: experts_per_tier;

% ============================================================================
% CONSTRAINTS
% ============================================================================

% Prices must be positive and increasing
constraint forall(i in 1..max_experts_per_review) (
    price_per_level[i] > 0
);

constraint forall(i in 1..max_experts_per_review-1) (
    price_per_level[i+1] > price_per_level[i]
);

% Customers must be non-negative
constraint forall(i in 1..max_experts_per_review) (
    customers_per_tier[i] >= 0
);

% Experts per tier must match tier level
constraint forall(i in 1..max_experts_per_review) (
    experts_per_tier[i] = i
);

% Total expert capacity constraint
constraint sum(i in 1..max_experts_per_review) (
    customers_per_tier[i] * experts_per_tier[i] * num_crud_cells
) <= num_experts * 30 * 8;  % experts × days/month × hours/day

% Price must cover expert costs plus margin
constraint forall(i in 1..max_experts_per_review) (
    price_per_level[i] >= experts_per_tier[i] * base_expert_cost * num_crud_cells * 2
);

% ============================================================================
% DERIVED VARIABLES
% ============================================================================

% Number of review combinations per tier
array[1..max_experts_per_review] of var int: combinations_per_tier;

constraint forall(i in 1..max_experts_per_review) (
    combinations_per_tier[i] = 
        if i == 1 then num_experts * num_crud_cells
        else if i == 2 then (num_experts * (num_experts - 1) div 2) * num_crud_cells
        else if i == 3 then (num_experts * (num_experts - 1) * (num_experts - 2) div 6) * num_crud_cells
        else if i == 4 then (num_experts * (num_experts - 1) * (num_experts - 2) * (num_experts - 3) div 24) * num_crud_cells
        else 0
        endif endif endif endif
);

% Revenue per tier
array[1..max_experts_per_review] of var int: revenue_per_tier;

constraint forall(i in 1..max_experts_per_review) (
    revenue_per_tier[i] = customers_per_tier[i] * price_per_level[i]
);

% Total monthly revenue
var int: total_monthly_revenue = sum(revenue_per_tier);

% Total annual revenue
var int: total_annual_revenue = total_monthly_revenue * 12;

% Expert utilization rate
var float: expert_utilization = 
    sum(i in 1..max_experts_per_review) (
        customers_per_tier[i] * experts_per_tier[i] * num_crud_cells
    ) / (num_experts * 30 * 8);

% ============================================================================
% OBJECTIVE
% ============================================================================

% Maximize total annual revenue
solve maximize total_annual_revenue;

% ============================================================================
% OUTPUT
% ============================================================================

output [
    "OPTIMAL PRICING AND ALLOCATION\n",
    "==============================\n\n",
    "Experts Available: ", show(num_experts), "\n",
    "CRUD Cells: ", show(num_crud_cells), "\n",
    "Base Expert Cost: $", show(base_expert_cost), "/review\n\n",
    
    "PRICING TIERS:\n"
] ++
[
    "Tier ", show(i), " (", show(i), " experts):\n",
    "  Price: $", show(price_per_level[i]), "/month\n",
    "  Customers: ", show(customers_per_tier[i]), "\n",
    "  Combinations: ", show(combinations_per_tier[i]), "\n",
    "  Revenue: $", show(revenue_per_tier[i]), "/month\n\n"
    | i in 1..max_experts_per_review
] ++
[
    "\nTOTAL REVENUE:\n",
    "Monthly: $", show(total_monthly_revenue), "\n",
    "Annual: $", show(total_annual_revenue), "\n\n",
    
    "EXPERT UTILIZATION:\n",
    "Rate: ", show(expert_utilization * 100), "%\n\n",
    
    "ZK HACKERS GOTTA EAT: ✅\n"
];
