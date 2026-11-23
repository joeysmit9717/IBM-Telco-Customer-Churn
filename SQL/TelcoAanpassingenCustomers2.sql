SELECT * FROM customers_ml
---SELECT * FROM customers_2

/*
CREATE VIEW customers_ml AS
SELECT
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
	"Multiple Lines",
	"(IS) No Internet Service",
	"(IS) Fiber Optics",
	"(IS) DSL",
	"(OS) Yes",
	"(OS) No",
	"(OS) No internet service",
	"Online Backup",
	"Device Protection",
	"Tech Support",
	"Streaming TV",
	"Streaming Movies", 
	"Contract",
	"Paperless Billing",
	"Payment Method",
	"Monthly Charges",
	"Total Charges",
	"Churn Value",
	"Churn Reason"	
FROM customers_2;
---DROP VIEW customers_ml
*/

/*
CREATE VIEW customers_ml AS
SELECT
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
	"Multiple Lines",
	"(IS) No Internet Service",
	"(IS) Fiber Optics",
	"(IS) DSL",
	"(OS) Yes",
	"(OS) No",
	"(OS) No internet service",
	"Online Backup",
	"Device Protection",
	"Tech Support",
	"Streaming TV",
	"Streaming Movies", 
	"Contract",
	"Paperless Billing",
	"Payment Method",
	"Monthly Charges",
	"Total Charges",
	"Churn Value",
	"Churn Score",
	"CLTV",
	"Churn Reason"	
FROM customers_2;
*/

/*
UPDATE customers_2
SET
    "(OS) Yes" =
        ("Online Security" = 'Yes')::INT,

    "(OS) No" =
        ("Online Security" = 'No')::INT,

    "(OS) No internet service" =
        ("Online Security" = 'No internet service')::INT;
*/
---ALTER TABLE customers_2
---ADD COLUMN "(OS) Yes" INTEGER,
---ADD COLUMN "(OS) No" INTEGER,
---ADD COLUMN "(OS) No internet service" INTEGER;

---Update customers_2
---SET "Multiple Lines" = REPLACE("Multiple Lines", '0 phone service', '0')

/*
UPDATE customers_2
SET
    "(IS) No Internet Service" =
        CASE WHEN "Internet Service" = 'No' THEN 1 ELSE 0 END,

    "(IS) Fiber Optics" =
        CASE WHEN "Internet Service" = 'Fiber optic' THEN 1 ELSE 0 END,

    "(IS) DSL" =
        CASE WHEN "Internet Service" = 'DSL' THEN 1 ELSE 0 END;
*/
		
---ALTER TABLE customers_2
---ADD "(IS) No Internet Service" INTEGER;
---ADD "(IS) Fiber Optics" INTEGER;
---ADD "(IS) DSL" INTEGER;

---ALTER TABLE customers_2
---DROP COLUMN "Churn Label";

---Update customers_2
---SET "Paperless Billing" = REPLACE("Paperless Billing", 'No', '0')
---SET "Paperless Billing" = REPLACE("Paperless Billing", 'Yes', '1')

---Update customers_2
---SET "Multiple Lines" = REPLACE("Multiple Lines", 'No', '0')
---SET "Multiple Lines" = REPLACE("Multiple Lines", 'Yes', '1')


---Update customers_2
---SET "Phone Service" = REPLACE("Phone Service", 'No', '0')
---SET "Phone Service" = REPLACE("Phone Service", 'Yes', '1')

---SELECT COUNT(DISTINCT "Internet Service") FROM customers_2
---SELECT COUNT(DISTINCT "City") FROM customers_2

---Update customers_2
---SET "Dependents" = REPLACE("Dependents", 'No', '0')
---SET "Dependents" = REPLACE("Dependents", 'Yes', '1')

---Update customers_2
---SET "Partner" = REPLACE("Partner", 'No', '0')
---SET "Partner" = REPLACE("Partner", 'Yes', '1')

---Senior Citizen     
---Update customers_2
---SET "Senior Citizen" = REPLACE("Senior Citizen", 'No', '0')
---SET "Senior Citizen" = REPLACE("Senior Citizen", 'Yes', '1')

/* remove male/female and change to 0/1
UPDATE customers_2
SET "Gender" = REPLACE("Gender", 'Female', '1');
*/
/*
ALTER TABLE customers_2
DROP COLUMN "Lat Long";
*/
/*
CREATE TABLE customers_2 AS
TABLE customers;
*/
/*
SELECT * FROM customers
WHERE "Churn Label" = 'Yes'
*/