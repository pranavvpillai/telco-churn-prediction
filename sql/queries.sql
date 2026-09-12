-- ============================================================
-- Churn Analytics -- SQL Query Set
-- Run against data/churn.db (SQLite). Schema: customers, services, billing
-- ============================================================


-- ----------------------------------------------------------------
-- 1. Overall churn rate + revenue at risk
--    (basic aggregation, join customers + billing)
-- ----------------------------------------------------------------
SELECT
    c.Churn,
    COUNT(*) AS customer_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 1) AS pct_of_total,
    ROUND(SUM(b.MonthlyCharges), 2) AS total_monthly_revenue
FROM customers c
JOIN billing b ON c.customerID = b.customerID
GROUP BY c.Churn;


-- ----------------------------------------------------------------
-- 2. Churn rate by contract type and internet service
--    (three-way join across all tables)
-- ----------------------------------------------------------------
SELECT
    c.Contract,
    s.InternetService,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN c.Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN c.Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers c
JOIN services s ON c.customerID = s.customerID
JOIN billing b ON c.customerID = b.customerID
GROUP BY c.Contract, s.InternetService
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------
-- 3. Churn rate by tenure cohort
--    (CTE bucketing tenure into cohorts, then aggregating)
-- ----------------------------------------------------------------
WITH tenure_cohorts AS (
    SELECT
        customerID,
        Churn,
        CASE
            WHEN tenure <= 12 THEN '0-12 months'
            WHEN tenure <= 24 THEN '13-24 months'
            WHEN tenure <= 48 THEN '25-48 months'
            ELSE '49+ months'
        END AS tenure_cohort
    FROM customers
)
SELECT
    tenure_cohort,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM tenure_cohorts
GROUP BY tenure_cohort
ORDER BY
    CASE tenure_cohort
        WHEN '0-12 months' THEN 1
        WHEN '13-24 months' THEN 2
        WHEN '25-48 months' THEN 3
        ELSE 4
    END;


-- ----------------------------------------------------------------
-- 4. Rank customers by monthly charges within their contract type
--    (window function: RANK() partitioned by Contract)
-- ----------------------------------------------------------------
SELECT
    c.customerID,
    c.Contract,
    b.MonthlyCharges,
    RANK() OVER (PARTITION BY c.Contract ORDER BY b.MonthlyCharges DESC) AS charge_rank_in_contract
FROM customers c
JOIN billing b ON c.customerID = b.customerID
ORDER BY c.Contract, charge_rank_in_contract
LIMIT 30;


-- ----------------------------------------------------------------
-- 5. Running cumulative revenue by tenure
--    (window function: SUM() OVER, ordered running total)
-- ----------------------------------------------------------------
SELECT
    tenure,
    COUNT(*) AS customers_at_this_tenure,
    SUM(SUM(b.MonthlyCharges)) OVER (ORDER BY tenure) AS cumulative_monthly_revenue
FROM customers c
JOIN billing b ON c.customerID = b.customerID
GROUP BY tenure
ORDER BY tenure;


-- ----------------------------------------------------------------
-- 6. Revenue at risk from churned customers, by payment method
--    (join + aggregation, business framing: "how much revenue is walking out the door")
-- ----------------------------------------------------------------
SELECT
    c.PaymentMethod,
    COUNT(*) AS churned_customers,
    ROUND(SUM(b.MonthlyCharges), 2) AS monthly_revenue_at_risk,
    ROUND(SUM(b.MonthlyCharges) * 12, 2) AS annualized_revenue_at_risk
FROM customers c
JOIN billing b ON c.customerID = b.customerID
WHERE c.Churn = 'Yes'
GROUP BY c.PaymentMethod
ORDER BY monthly_revenue_at_risk DESC;


-- ----------------------------------------------------------------
-- 7. Data quality audit: referential integrity check
--    (every customerID in customers should exist in services and billing)
-- ----------------------------------------------------------------
SELECT 'services missing customers' AS check_name, COUNT(*) AS violation_count
FROM customers c
LEFT JOIN services s ON c.customerID = s.customerID
WHERE s.customerID IS NULL

UNION ALL

SELECT 'billing missing customers', COUNT(*)
FROM customers c
LEFT JOIN billing b ON c.customerID = b.customerID
WHERE b.customerID IS NULL

UNION ALL

SELECT 'duplicate customerIDs', COUNT(*) - COUNT(DISTINCT customerID)
FROM customers

UNION ALL

SELECT 'null TotalCharges (expected only for tenure=0)', COUNT(*)
FROM billing b
JOIN customers c ON b.customerID = c.customerID
WHERE b.TotalCharges IS NULL AND c.tenure > 0;
