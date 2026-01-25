"""
Knowledge Base Population Script

This script loads drug information from JSON files and populates
the ChromaDB knowledge base for RAG retrieval.

Usage:
    python scripts/populate_knowledge_base.py
    python scripts/populate_knowledge_base.py --clear  # Clear existing data first
    python scripts/populate_knowledge_base.py --file custom_drugs.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.rag import ChromaKnowledgeRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def format_drug_document(drug_data: dict) -> str:
    """
    Format drug data into a comprehensive text document for embedding.
    
    This creates a rich text representation that will be embedded
    and used for semantic search.
    """
    parts = []
    
    # Drug name and basic info
    drug_name = drug_data.get("drug_name", "Unknown")
    parts.append(f"Drug Name: {drug_name}")
    
    # Active ingredients
    if ingredients := drug_data.get("active_ingredients"):
        parts.append(f"Active Ingredients: {', '.join(ingredients)}")
    
    # Manufacturer
    if manufacturer := drug_data.get("manufacturer"):
        parts.append(f"Manufacturer: {manufacturer}")
    
    # Dosage form and strengths
    if dosage_form := drug_data.get("dosage_form"):
        parts.append(f"Dosage Form: {dosage_form}")
    
    if strengths := drug_data.get("strengths"):
        parts.append(f"Available Strengths: {', '.join(strengths)}")
    
    # Description
    if description := drug_data.get("description"):
        parts.append(f"\nDescription: {description}")
    
    # Indications (what it's used for)
    if indications := drug_data.get("indications"):
        parts.append(f"\nIndications: {indications}")
    
    # Usage instructions
    if usage := drug_data.get("usage"):
        parts.append(f"\nUsage: {usage}")
    
    # Warnings
    if warnings := drug_data.get("warnings"):
        parts.append("\nWarnings:")
        for warning in warnings:
            parts.append(f"- {warning}")
    
    # Side effects
    if side_effects := drug_data.get("side_effects"):
        parts.append("\nPossible Side Effects:")
        for effect in side_effects:
            parts.append(f"- {effect}")
    
    # Drug interactions
    if interactions := drug_data.get("interactions"):
        parts.append("\nDrug Interactions:")
        for interaction in interactions:
            parts.append(f"- {interaction}")
    
    # Contraindications
    if contraindications := drug_data.get("contraindications"):
        parts.append("\nContraindications:")
        for contra in contraindications:
            parts.append(f"- {contra}")
    
    # Storage
    if storage := drug_data.get("storage"):
        parts.append(f"\nStorage: {storage}")
    
    # Pregnancy category
    if pregnancy := drug_data.get("pregnancy_category"):
        parts.append(f"\nPregnancy Category: {pregnancy}")
    
    # Prescription status
    if prescription := drug_data.get("prescription_status"):
        parts.append(f"\nPrescription Status: {prescription}")
    
    return "\n".join(parts)


def load_drugs_from_json(file_path: Path) -> list:
    """Load drug data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            logger.error(f"Invalid JSON format in {file_path}")
            return []
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def populate_knowledge_base(
    json_file: Path,
    persist_directory: str = "./data/chroma_db",
    clear_existing: bool = False
):
    """
    Populate the knowledge base with drug information.
    
    Args:
        json_file: Path to JSON file with drug data
        persist_directory: ChromaDB persistence directory
        clear_existing: Whether to clear existing data first
    """
    logger.info(f"Loading drug data from: {json_file}")
    
    # Load drug data
    drugs = load_drugs_from_json(json_file)
    
    if not drugs:
        logger.error("No drug data loaded")
        return
    
    logger.info(f"Loaded {len(drugs)} drug(s)")
    
    # Initialize ChromaDB retriever
    logger.info(f"Initializing ChromaDB (persist_directory={persist_directory})")
    retriever = ChromaKnowledgeRetriever(
        persist_directory=persist_directory,
        collection_name="drug_knowledge"
    )
    
    # Clear existing data if requested
    if clear_existing:
        logger.warning("Clearing existing knowledge base...")
        retriever.clear()
    
    # Check current size
    current_size = retriever.knowledge_base_size
    logger.info(f"Current knowledge base size: {current_size} documents")
    
    # Index each drug
    success_count = 0
    for i, drug in enumerate(drugs, 1):
        drug_name = drug.get("drug_name", f"Unknown_{i}")
        logger.info(f"Processing {i}/{len(drugs)}: {drug_name}")
        
        # Format drug data into document
        content = format_drug_document(drug)
        
        # Prepare metadata
        metadata = {
            "id": f"drug_{drug_name.lower().replace(' ', '_')}",
            "drug_name": drug_name,
            "source": str(json_file),
            "type": "drug_information"
        }
        
        # Add manufacturer and ingredients to metadata for filtering
        if manufacturer := drug.get("manufacturer"):
            metadata["manufacturer"] = manufacturer
        
        if ingredients := drug.get("active_ingredients"):
            metadata["active_ingredients"] = ", ".join(ingredients)
        
        # Index the document
        if retriever.index_document(content, metadata):
            success_count += 1
            logger.info(f"✓ Successfully indexed: {drug_name}")
        else:
            logger.error(f"✗ Failed to index: {drug_name}")
    
    # Final summary
    new_size = retriever.knowledge_base_size
    logger.info("=" * 60)
    logger.info(f"Population complete!")
    logger.info(f"Successfully indexed: {success_count}/{len(drugs)} drugs")
    logger.info(f"Knowledge base size: {current_size} → {new_size} documents")
    logger.info("=" * 60)
    
    # Test query
    if success_count > 0:
        logger.info("\nTesting retrieval...")
        test_drug = drugs[0].get("drug_name", "")
        if test_drug:
            result = retriever.retrieve_by_drug_name(test_drug, top_k=1)
            if result.chunks:
                logger.info(f"✓ Test query for '{test_drug}' successful")
                logger.info(f"  Relevance score: {result.chunks[0].relevance_score:.2f}")
            else:
                logger.warning(f"✗ Test query for '{test_drug}' returned no results")


def main():
    parser = argparse.ArgumentParser(
        description="Populate drug knowledge base from JSON files"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default="data/drug_knowledge_base/drugs.json",
        help="Path to JSON file with drug data"
    )
    
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./data/chroma_db",
        help="ChromaDB persistence directory"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing knowledge base before populating"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    json_file = Path(args.file)
    if not json_file.exists():
        logger.error(f"File not found: {json_file}")
        logger.info(f"Please create the file or specify a different path with --file")
        sys.exit(1)
    
    # Run population
    try:
        populate_knowledge_base(
            json_file=json_file,
            persist_directory=args.persist_dir,
            clear_existing=args.clear
        )
    except Exception as e:
        logger.error(f"Population failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
