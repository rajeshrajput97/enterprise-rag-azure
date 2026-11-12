# Contoso Cloud Records Policy (sample)

This is a fictional policy used only as sample input for the RAG reference implementation.

## Retention

Customer transaction records must be retained for seven (7) years from the date of the transaction.
Security logs are retained for 18 months. Draft working documents may be deleted after 90 days
if they have not been declared as records.

## Access

Access to production data is granted on a least-privilege basis. Privileged access reviews
are completed quarterly. Temporary elevated access expires automatically after 24 hours.

## Incident response

Severity-1 incidents must be acknowledged within 15 minutes and have a customer-facing
update within one hour. A written post-incident review is required within five business days.

## Data residency

Personal data of customers in the European Union is stored in the West Europe Azure region.
Personal data of customers in India is stored in the Central India Azure region unless a
written exception is approved by the data protection officer.
