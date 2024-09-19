# NovelAI ATTG Exploration Tool

This project aims to develop a tool for exploring and optimizing the use of ATTG (Author, Title, Tags, Genre) in NovelAI's text generation system. The goal is to automate the process of finding effective terms for each ATTG category and analyze their impact on text generation.

## Background
NovelAI is a GPT-based service for creative writing. It uses models trained in-house by Anlatan, focusing on prose completion rather than chat-based interactions. The ATTG system is used to steer the AI's output by providing context through meta-tags.

## Project Goals
1. Generate candidate terms for each ATTG category
2. Test the effectiveness of these terms
3. Analyze the impact of different terms on text generation
4. Provide insights for optimal ATTG usage in NovelAI

## Methodology
1. Generate candidate terms using the NovelAI API
2. Test each term multiple times and compare with control prompts
3. Analyze results using log odds and statistical tests
4. Determine significance and effect size for each term