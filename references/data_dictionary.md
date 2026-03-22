# TalentLens Data Dictionary

## Data Source
LinkedIn Job Postings dataset from Kaggle (2023-2024)

---

## Main Table: `postings.csv` (3.38M rows)

| Column | Type | Description | Null Rate |
|--------|------|-------------|-----------|
| job_id | int64 | Unique identifier for each job posting | 0% |
| company_name | str | Name of the hiring company | Low |
| title | str | Job title | Very low |
| description | str | Full text of the job posting | Low |
| max_salary | float | Maximum salary offered | High |
| med_salary | float | Median salary offered | High |
| min_salary | float | Minimum salary offered | High |
| pay_period | str | YEARLY or HOURLY | High |
| location | str | City, State format | Low |
| company_id | float | Foreign key to companies table | Low |
| views | float | Number of views on LinkedIn | Medium |
| applies | float | Number of applications received | Medium |
| original_listed_time | float | Unix timestamp (ms) when first posted | Low |
| listed_time | float | Unix timestamp (ms) of current listing | Low |
| closed_time | float | Unix timestamp (ms) when posting closed | Medium |
| remote_allowed | float | 1.0 if remote, NaN if not | High (most are NaN = not remote) |
| formatted_experience_level | str | Entry level, Mid-Senior level, etc. | Medium |
| skills_desc | str | Comma-separated skills text | Medium |
| formatted_work_type | str | Full-time, Part-time, Contract, etc. | Low |
| sponsored | float | Whether the posting is sponsored | Medium |
| work_type | str | Internal work type code | Low |
| currency | str | Currency code (usually USD) | Medium |
| compensation_type | str | BASE_SALARY, etc. | Medium |
| normalized_salary | float | Normalized salary value | High |
| job_posting_url | str | URL to the posting | Low |
| application_url | str | URL to apply | Medium |
| application_type | str | Type of application process | Medium |
| expiry | float | Unix timestamp (ms) of expiry | Medium |
| posting_domain | str | Domain where the job was posted | Low |
| zip_code | str | ZIP code of job location | Medium |
| fips | float | FIPS code for geographic region | High |

### Derived Columns (after cleaning)
| Column | Type | Description |
|--------|------|-------------|
| med_salary_yearly | float | Median salary normalized to yearly (hourly x 2080) |
| min_salary_yearly | float | Minimum salary normalized to yearly |
| max_salary_yearly | float | Maximum salary normalized to yearly |
| is_remote | bool | True if remote_allowed = 1.0 |
| days_open | float | Number of days between listing and closing |
| experience_level | str | Cleaned version of formatted_experience_level |

---

## Secondary Tables

### `companies/companies.csv` (141K rows)
| Column | Description |
|--------|-------------|
| company_id | Unique company identifier |
| name | Company name |
| description | Company description text |
| company_size | Size category (0-7 scale) |
| state | US state |
| country | Country code |
| city | City |
| zip_code | ZIP code |
| address | Street address |
| url | LinkedIn company page URL |

### `companies/employee_counts.csv` (36K rows)
| Column | Description |
|--------|-------------|
| company_id | Foreign key to companies |
| employee_count | Number of employees |
| follower_count | LinkedIn follower count |
| time_recorded | Unix timestamp of the count |

### `companies/company_industries.csv` (24K rows)
| Column | Description |
|--------|-------------|
| company_id | Foreign key to companies |
| industry | Industry name string |

### `companies/company_specialities.csv` (169K rows)
| Column | Description |
|--------|-------------|
| company_id | Foreign key to companies |
| speciality | Company specialization |

### `jobs/benefits.csv` (68K rows)
| Column | Description |
|--------|-------------|
| job_id | Foreign key to postings |
| inferred | Whether the benefit was inferred |
| type | Benefit type (Medical, Dental, 401k, etc.) |

### `jobs/job_skills.csv` (214K rows)
| Column | Description |
|--------|-------------|
| job_id | Foreign key to postings |
| skill_abr | Skill abbreviation (e.g., IT, SALE, DSGN) |

### `jobs/job_industries.csv` (165K rows)
| Column | Description |
|--------|-------------|
| job_id | Foreign key to postings |
| industry_id | Foreign key to industries mapping |

### `jobs/salaries.csv` (41K rows)
| Column | Description |
|--------|-------------|
| salary_id | Unique salary record ID |
| job_id | Foreign key to postings |
| max_salary | Maximum salary |
| med_salary | Median salary |
| min_salary | Minimum salary |
| pay_period | YEARLY or HOURLY |
| currency | Currency code |
| compensation_type | BASE_SALARY, etc. |

---

## Mapping / Lookup Tables

### `mappings/skills.csv` (35 rows)
| skill_abr | skill_name |
|-----------|------------|
| IT | Information Technology |
| SALE | Sales |
| DSGN | Design |
| MRKT | Marketing |
| ... | (35 total categories) |

### `mappings/industries.csv` (422 rows)
| industry_id | industry_name |
|-------------|---------------|
| 1 | Defense and Space Manufacturing |
| 3 | Computer Hardware Manufacturing |
| 4 | Software Development |
| ... | (422 total industries) |
