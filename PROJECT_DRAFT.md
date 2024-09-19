# Detailed Workflow: NovelAI ATTG Exploration Tool

## Overview
This tool aims to systematically explore and optimize the use of ATTG (Author, Title, Tags, Genre) in NovelAI's text generation system. The process is divided into four main phases: Candidate Generation, Term Testing, Statistical Analysis, and Results Presentation.

## NovelAI and ATTG System Explained

NovelAI is a platform that uses advanced language models for creative writing assistance. Unlike chatbots or general-purpose AI assistants, NovelAI specializes in continuing and generating prose based on user input.

The ATTG system is a unique feature of NovelAI that allows users to guide the AI's output:

- Author: Can influence the writing style or voice.
- Title: May set the tone or theme of the generated text.
- Tags: Provide specific elements or themes to include.
- Genre: Sets the overall category and conventions of the text.

ATTG is typically used in the format:
[ Author: ; Title: ; Tags: ; Genre: ]

For example:
[ Author: Jane Austen; Title: A Modern Romance; Tags: technology, social media; Genre: romantic comedy ]

This project aims to help NovelAI users find the most effective terms for each ATTG category, potentially improving the quality and relevance of AI-generated text for their specific needs.

## Phase 1: Candidate Generation

1. Initial Prompting:
   - For each ATTG category (Author, Title, Tags, Genre), the tool prompts the NovelAI API with an incomplete tag (e.g., `[ Author:` or `[ Tags:`).
   - The API's completions are recorded as candidate terms for each category.

2. Bias Management:
   - A bias system is implemented to encourage diversity in generated terms.
   - When a term is generated multiple times, its bias is adjusted to reduce its likelihood in future generations.

3. Storage:
   - Unique terms are stored in a database or structured file system, along with metadata such as generation frequency and current bias value.

## Phase 2: Term Testing

1. Prompt Creation:
   - For each candidate term, the tool creates multiple prompts (e.g., 20 per term) using the full ATTG format.
   - Example: `[ Author: Shakespeare ]`

2. Control Prompts:
   - The tool also creates control prompts with empty ATTG tags (e.g., `[ Author: ]`).

3. Text Generation:
   - Using the NovelAI API, the tool generates text completions for each prompt (including control prompts).
   - All generated texts are stored, along with their corresponding prompts.

## Phase 3: Statistical Analysis

1. Data Preparation:
   - For each generated text, the tool removes the original prompt.
   - A new prompt line is added (e.g., `[ Author:`) to standardize the context.

2. Token Analysis:
   - The tool calculates the log odds of the token from the original term appearing in the generated text.
   - This data is recorded along with the originating term.

3. Statistical Tests:
   - The tool performs significance tests comparing results from each term to the control prompts.
   - Effect sizes are computed for each term to quantify its impact on text generation.

4. Results Compilation:
   - For each term, the tool records:
     a) The effect size
     b) Statistical significance
     c) Frequency of appearance in generated texts
     d) Any notable patterns or anomalies

## Phase 4: Results Presentation

1. Data Visualization:
   - The tool creates visualizations (e.g., bar charts, heatmaps) to illustrate the effectiveness of different terms within each ATTG category.

2. Ranking System:
   - Terms are ranked within their categories based on their effect sizes and statistical significance.

3. Insights Generation:
   - The tool provides written insights about the most effective terms, unexpected results, and potential strategies for ATTG usage.

4. User Interface:
   - A user-friendly interface allows users to:
     a) View top-performing terms for each ATTG category
     b) Explore detailed statistics for individual terms
     c) Generate custom ATTG combinations based on their specific needs
     d) Potentially test their own custom terms

5. Feedback Loop:
   - Users can provide feedback on the usefulness of suggested terms or combinations.
   - This feedback is used to refine the tool's recommendations over time.

## Continuous Improvement

1. Regular Re-testing:
   - The tool periodically re-tests terms to account for potential changes in the NovelAI model's behavior over time.

2. Expansion of Term Database:
   - The tool continues to generate and test new terms, gradually expanding its database of analyzed ATTG components.

3. Advanced Analysis:
   - As the dataset grows, more advanced analyses can be implemented, such as:
     a) Identifying synergies between terms in different ATTG categories
     b) Analyzing the impact of term combinations on specific genres or writing styles
     c) Developing predictive models for optimal ATTG combinations based on user inputs

By following this workflow, the NovelAI ATTG Exploration Tool will provide users with data-driven insights for optimizing their use of the ATTG system, potentially enhancing the quality and relevance of AI-generated text for their creative writing projects.