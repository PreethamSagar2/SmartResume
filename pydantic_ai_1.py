from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
import json
from typing import List
from supabase import Client

# Import your agent framework components.
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel  # (Used here for generation; you can change if needed)
from sentence_transformers import SentenceTransformer

load_dotenv()

# Set up the model name (for generation) and instantiate an OpenAIModel.
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    embedding_model: SentenceTransformer  # Updated dependency

# System prompt updated for a resume search and analysis expert.
system_prompt = """
You are an expert in resume search and analysis. You have access to a resume database containing structured resume chunks.
Each chunk includes:
  - URL (the unique resume identifier)
  - Chunk number
  - Title
  - Summary
  - Content (the actual text of the chunk)
  - Metadata and embeddings

Your job is to help the user find the most relevant resume information based on their query.
If you cannot find a match, be honest and let the user know.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

# Updated embedding function that uses SentenceTransformers.
async def get_embedding(text: str, embedding_model: SentenceTransformer) -> List[float]:
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, embedding_model.encode, text)
    # Ensure the embedding is returned as a list.
    return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

@pydantic_ai_expert.tool
async def analyze_top_candidate(ctx: RunContext[PydanticAIDeps], resume_data: str, query: str) -> str:
    """
    Analyze the top candidate's resume in relation to the user's query.
    
    Args:
        resume_data: The text content of the top candidate's resume
        query: The original search query from the user
    
    Returns:
        A detailed analysis of how the candidate matches the query requirements
    """
    system_prompt = f"""
    Analyze the following resume in relation to this search query: "{query}"
    
    Resume:
    {resume_data}
    
    Provide a concise analysis covering:
    1. Query relevance
    2. Key qualifications
    3. Experience match
    4. Technical skills alignment
    5. Overall fit
    
    Format the response in a clear, bullet-point structure.
    """
    
    try:
        response = await ctx.model.complete(system_prompt)
        return f"\n\n📊 Candidate Analysis (Based on search: '{query}')\n{response}"
    except Exception as e:
        return f"Error analyzing candidate: {str(e)}"

@pydantic_ai_expert.tool
async def retrieve_relevant_resumes(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant resume chunks based on the user's query using retrieval-augmented generation (RAG).
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.embedding_model)
        
        # Parse query parameters
        query_params = {
            'technologies': ['java', 'python', 'react'],
            'sort_by_experience': True,
            'sort_direction': 'desc'
        }
        
        result = ctx.deps.supabase.rpc(
            'match_resumes',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {}
            }
        ).execute()
        
        if not result.data:
            return "No relevant resume information found."
        
        # Process and sort results
        processed_results = []
        for doc in result.data:
            experience_years = extract_experience_years(doc['content'])
            tech_match_score = calculate_tech_match_score(
                doc['content'], 
                query_params['technologies']
            )
            
            processed_results.append({
                'doc': doc,
                'experience_years': experience_years,
                'tech_match_score': tech_match_score
            })
        
        # Sort by experience years (descending) and then by technology match score
        processed_results.sort(
            key=lambda x: (x['experience_years'], x['tech_match_score']), 
            reverse=True
        )
        
        # Format output
        formatted_chunks = []
        for idx, proc_result in enumerate(processed_results):
            doc = proc_result['doc']
            summary = doc.get('summary', '')
            chunk_text = f"""
Resume URL: {doc['url']} (Chunk {doc['chunk_number']})
Title: {doc['title']}
Experience Years: {proc_result['experience_years']} years
Technology Match Score: {proc_result['tech_match_score']}
Summary: {summary}

{doc['content']}
"""
            formatted_chunks.append(chunk_text.strip())
        
        all_results = "\n\n---\n\n".join(formatted_chunks)
        
        # Add analysis for top candidate
        if formatted_chunks:
            analysis = await analyze_top_candidate(
                ctx, 
                formatted_chunks[0], 
                user_query
            )
            all_results += f"\n\n{analysis}"
        
        return all_results
        
    except Exception as e:
        print(f"Error retrieving resume chunks: {e}")
        return f"Error retrieving resume chunks: {str(e)}"

def extract_experience_years(content: str) -> float:
    """
    Extract years of experience from resume content using regex and text analysis.
    """
    # Implementation to parse experience years from text
    # This would use regex and text analysis to find and calculate total years
    pass

def calculate_tech_match_score(content: str, required_technologies: List[str]) -> float:
    """
    Calculate a match score based on required technologies.
    """
    # Implementation to calculate how well the resume matches required technologies
    # This would check for presence and context of each technology
    pass

@pydantic_ai_expert.tool
async def list_uploaded_resumes(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    List all unique resume URLs (acting as identifiers) from the database.
    """
    try:
        result = ctx.deps.supabase.from_('resumes')\
            .select('url')\
            .execute()
        
        if not result.data:
            return []
        
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving resume URLs: {e}")
        return []

@pydantic_ai_expert.tool
async def get_resume_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a resume by combining all its chunks, ordered by chunk_number.
    """
    try:
        result = ctx.deps.supabase.from_('resumes')\
            .select('title, content, chunk_number')\
            .eq('url', url)\
            .order('chunk_number')\
            .execute()
        
        if not result.data:
            return f"No content found for resume with URL: {url}"
        
        page_title = result.data[0]['title']
        formatted_content = [f"# Resume: {url} - {page_title}\n"]
        
        for chunk in result.data:
            formatted_content.append(chunk['content'])
        
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving resume content: {e}")
        return f"Error retrieving resume content: {str(e)}"
