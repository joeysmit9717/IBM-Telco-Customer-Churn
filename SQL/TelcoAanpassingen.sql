SELECT * FROM customers;
/*
ALTER TABLE customers
ALTER COLUMN "Total Charges" TYPE NUMERIC(8,2)
USING "Total Charges"::NUMERIC(8,2);
*/

/*
SELECT * FROM customers
WHERE "Churn Label" = 'No';
*/

/*
UPDATE customers
SET "Total Charges" = '0.00'
WHERE TRIM("Total Charges") = '';
*/

/*
SELECT
    "CustomerID",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges"
FROM customers 
WHERE "Total Charges" = '0.00';
*/

/*
WHERE "Total Charges" = ' ';
*/