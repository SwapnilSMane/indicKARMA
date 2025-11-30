import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import aiohttp
from cachetools import TTLCache
from asyncache import cached
from cachetools.keys import hashkey
from src.resource_manager import register_session, register_connector

DEFAULT_CONTEXT_CONFIG = {
    "sources": {
        "conceptnet": {
            "enabled": True,
            "api_endpoint": "http://api.conceptnet.io/c/en/",
            "timeout": 10.0,
            "cache_ttl": 7200,
            "max_errors": 5,
            "error_cooldown": 300,
        },
        "wikidata": {
            "enabled": True,
            "sparql_endpoint": "https://query.wikidata.org/sparql",
            "timeout": 10.0,
            "cache_ttl": 7200,
            "max_retries": 2,
            "user_agent": "ContextRetriever/1.0 (Python/aiohttp; Contact: your-email@example.com)",
        }
    },
    "cache_config": {
        "max_size": 1000,
        "ttl": 7200,
    },
    "agent_context_types": {
        "aggression_level": ["linguistic", "aggression", "social"],
        "aggression_type": ["linguistic", "aggression", "social"],
        "discursive_role": ["social", "linguistic"],
        "gender_bias": ["cultural", "social"],
        "religious_bias": ["cultural", "social"],
        "caste_bias": ["cultural", "social"],
        "ethnicity_bias": ["cultural", "social"],
        "graph_rag": ["linguistic", "conceptual", "social", "entity"],
        "explainer": ["cultural", "social", "conceptual", "entity"]
    },
    "settings": {
        "context_timeout": 5.0,
        "enable_sharing": True,
        "max_context_items": 10,
        "parallel_queries": True,
        "fallback_to_local": False,
        "cache_preload": False,
    }
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContextRetriever:

    def __init__(self, config: Dict = DEFAULT_CONTEXT_CONFIG):
        self.config = config
        self.sources_config = config.get("sources", {})
        self.agent_context_types = config.get("agent_context_types", {})
        self.settings = config.get("settings", {})

        self.context_timeout = self.settings.get("context_timeout", 5.0)
        self.enable_sharing = self.settings.get("enable_sharing", True)
        self.max_context_items = self.settings.get("max_context_items", 10)
        self.parallel_queries = self.settings.get("parallel_queries", True)
        self.fallback_to_local = self.settings.get("fallback_to_local", False)

        self.timeout = max(
        self.sources_config.get("conceptnet", {}).get("timeout", 10.0),
        self.sources_config.get("wikidata", {}).get("timeout", 10.0)
        )

        cache_conf = config.get("cache_config", {})
        self.cache = TTLCache(maxsize=cache_conf.get("max_size", 1000),
                              ttl=cache_conf.get("ttl", 7200))

        self.metrics = {}
        self.reset_metrics()

        self._source_state = {
            source_name: {"error_count": 0, "last_error_time": 0}
            for source_name in self.sources_config
        }

        self.session: Optional[aiohttp.ClientSession] = None
        self._session_created_internally = False
        self.initialized = False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            logging.info("Creating internal aiohttp client session.")
            self.session = aiohttp.ClientSession()
            self._session_created_internally = True
        return self.session
    
    async def initialize(self):
        connector = aiohttp.TCPConnector(limit=5, ttl_dns_cache=300, ssl=False)
        connector = register_connector(connector)
        
        self.session = register_session(aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=connector,
            headers={'User-Agent': 'KG-Agent/1.0'}
        ))
        self.initialized = True

    def reset_metrics(self) -> None:
        self.metrics = {
            "total_queries_requested": 0,
            "cache_hits": 0,
            "retrievals_attempted": 0,
            "retrievals_succeeded": 0,
            "retrievals_failed": 0,
            "source_success": {src: 0 for src in self.sources_config},
            "source_failure": {src: 0 for src in self.sources_config},
            "fallback_used_count": 0,
            "total_retrieval_time_ms": 0,
            "avg_retrieval_time_ms": 0,
        }

    def get_metrics(self) -> Dict[str, Any]:
        if self.metrics["retrievals_succeeded"] > 0:
             self.metrics["avg_retrieval_time_ms"] = round(
                 self.metrics["total_retrieval_time_ms"] / self.metrics["retrievals_succeeded"]
             )
        else:
             self.metrics["avg_retrieval_time_ms"] = 0
        return self.metrics

    async def get_context(self, term: str, agent_type: str) -> Dict[str, Any]:
        logging.info(f"Context requested for term='{term}', agent='{agent_type}'")
        self.metrics["total_queries_requested"] += 1
        cache_key = hashkey(term)
        try:
            cached_context = self.cache[cache_key]
            logging.info(f"Cache hit for term='{term}'.")
            self.metrics["cache_hits"] += 1
            retrieved_context = cached_context
        except KeyError:
            logging.info(f"Cache miss for term='{term}'. Querying sources...")
            self.metrics["retrievals_attempted"] += 1
            start_time = time.monotonic()
            try:
                raw_results = await self._query_sources(term)
                processed_context = self._process_results(raw_results, term, "base")
                end_time = time.monotonic()
                retrieval_time_ms = (end_time - start_time) * 1000
                self.metrics["retrievals_succeeded"] += 1
                self.metrics["total_retrieval_time_ms"] += retrieval_time_ms
                self.cache[cache_key] = processed_context
                retrieved_context = processed_context
            except Exception as e:
                logging.error(f"Failed to retrieve/process context for term='{term}': {e}", exc_info=True)
                self.metrics["retrievals_failed"] += 1
                retrieved_context = {
                    "term": term, "agent_type": agent_type, "items": [],
                    "summary": "Error retrieving context.", "sources_used": [],
                    "timestamp": time.time(), "error": str(e), "cache_hit": False
                }
        if 'cache_hit' not in retrieved_context:
             retrieved_context['cache_hit'] = (cache_key in self.cache and retrieved_context is self.cache[cache_key])
        final_context = retrieved_context
        if "error" not in final_context:
            current_agent_type = final_context.get("agent_type", "base")
            if self.enable_sharing and current_agent_type != agent_type:
                logging.debug(f"Filtering context for agent '{agent_type}' from type '{current_agent_type}'")
                if "items" in final_context:
                    final_context = self._filter_context_for_agent(final_context, agent_type)
                else:
                    logging.warning("Retrieved context missing 'items', cannot filter. Returning as is.")
                    final_context["agent_type"] = agent_type
            else:
                 final_context["agent_type"] = agent_type
        return final_context
    

    async def _query_sources(self, term: str) -> List[Dict[str, Any]]:
        tasks = []
        enabled_sources = []

        if not await self.ensure_session():
            logging.error("Failed to ensure session for context retrieval")
            return []

        for source_name, config in self.sources_config.items():
            if config.get("enabled"):
                if source_name == "conceptnet":
                    state = self._source_state[source_name]
                    cooldown = config.get("error_cooldown", 300)
                    if state["error_count"] >= config.get("max_errors", 5) and \
                    time.time() - state["last_error_time"] < cooldown:
                        logging.warning(f"ConceptNet source is in cooldown for term '{term}'. Skipping.")
                        continue

                enabled_sources.append(source_name)
                query_func = getattr(self, f"_query_{source_name}", None)
                if query_func:
                    tasks.append(query_func(term, config, self.session))
                else:
                    logging.warning(f"No query function found for enabled source: {source_name}")

        if not tasks:
            logging.warning(f"No enabled sources or query functions available for term '{term}'")
            return []
        
        all_results = []
        try:
            if self.parallel_queries:
                logging.debug(f"Running queries in parallel for term '{term}'")
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                logging.debug(f"Running queries sequentially for term '{term}'")
                results_list = []
                for task in tasks:
                    try:
                        results_list.append(await task)
                    except Exception as e:
                        results_list.append(e)

            for i, result in enumerate(results_list):
                source_name = enabled_sources[i]
                if isinstance(result, Exception):
                    logging.error(f"Error querying {source_name} for '{term}': {result}")
                    self.metrics["source_failure"][source_name] = self.metrics["source_failure"].get(source_name, 0) + 1
                    if source_name == "conceptnet":
                        self._source_state[source_name]["error_count"] += 1
                        self._source_state[source_name]["last_error_time"] = time.time()

                    if self.fallback_to_local:
                        logging.warning(f"Remote source {source_name} failed. Fallback requested (Not Implemented).")
                        self.metrics["fallback_used_count"] += 1

                elif isinstance(result, list):
                    logging.debug(f"Successfully received {len(result)} results from {source_name} for '{term}'")
                    all_results.extend(result)
                    self.metrics["source_success"][source_name] = self.metrics["source_success"].get(source_name, 0) + 1
                    if source_name == "conceptnet":
                       self._source_state[source_name]["error_count"] = 0
                else:
                     logging.warning(f"Unexpected result type from {source_name} for '{term}': {type(result)}")
                     self.metrics["source_failure"][source_name] = self.metrics["source_failure"].get(source_name, 0) + 1

        except Exception as e:
            logging.error(f"General error during source querying for '{term}': {e}", exc_info=True)

        return all_results

    async def _query_conceptnet(self, term: str, config: Dict, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        if not await self.ensure_session():
            logging.error("Cannot query ConceptNet: session unavailable")
            return []
        
        results = []
        normalized_term = term.lower().replace(" ", "_")
        url = f"{config['api_endpoint']}{normalized_term}"
        params = {'limit': 20}
        timeout = aiohttp.ClientTimeout(total=config.get('timeout', 2.0))

        logging.debug(f"Querying ConceptNet: {url}")
        try:
             async with self.session.get(url, params=params, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
                logging.debug(f"ConceptNet response status: {response.status} for {term}")

                for edge in data.get("edges", []):

                    start_uri = edge.get("start", {}).get("@id", "")
                    end_uri = edge.get("end", {}).get("@id", "")
                    
                    if not (start_uri.startswith("/c/en/") and end_uri.startswith("/c/en/")):
                        continue

                    start_node = edge.get("start", {}).get("label")
                    end_node = edge.get("end", {}).get("label")
                    relation = edge.get("rel", {}).get("label")
                    weight = edge.get("weight", 0.0)
                    surface_text = edge.get("surfaceText")

                    context_type = "linguistic" if relation in ["Synonym", "Antonym", "RelatedTo"] else "conceptual"

                    if start_node and end_node and relation:
                         if edge.get("start", {}).get("term", "").endswith(normalized_term):
                            results.append({
                                "source": start_node,
                                "relation": relation,
                                "target": end_node,
                                "weight": weight,
                                "explanation": surface_text or f"{start_node} {relation} {end_node}",
                                "source_name": "conceptnet",
                                "context_type": context_type,
                                "is_fallback": False,
                            })
                         elif edge.get("end", {}).get("term", "").endswith(normalized_term):
                             results.append({
                                "source": end_node,
                                "relation": f"inverse_{relation}",
                                "target": start_node,
                                "weight": weight * 0.8,
                                "explanation": surface_text or f"{start_node} {relation} {end_node}",
                                "source_name": "conceptnet",
                                "context_type": context_type,
                                "is_fallback": False,
                            })

        except aiohttp.ClientError as e:
            logging.error(f"ConceptNet HTTP Error for '{term}': {e}")
            raise
        except asyncio.TimeoutError:
            logging.error(f"ConceptNet Timeout for '{term}' after {config.get('timeout', 2.0)}s")
            raise
        except json.JSONDecodeError as e:
             logging.error(f"ConceptNet JSON Decode Error for '{term}': {e}")
             raise
        return results

    async def _query_wikidata(self, term: str, config: Dict, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        if not await self.ensure_session():
            logging.error("Cannot query Wikidata: session unavailable")
            return []
        
        results = []
        sparql_query = f"""
        SELECT ?item ?itemLabel ?itemDescription ?propLabel ?valueLabel ?valueDescription WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:api "EntitySearch" .
            bd:serviceParam wikibase:endpoint "www.wikidata.org" .
            bd:serviceParam mwapi:search "{term}" .
            bd:serviceParam mwapi:language "en" .
            ?item wikibase:apiOutputItem mwapi:item .
          }}

          OPTIONAL {{ ?item wdt:P31 ?valueP31 . }}
          OPTIONAL {{ ?item wdt:P106 ?valueP106 . }}

          BIND(COALESCE(?valueP31, ?valueP106) AS ?value)

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT 15
        """

        endpoint = config['sparql_endpoint']
        timeout = aiohttp.ClientTimeout(total=config.get('timeout', 10.0))
        headers = {
            'Accept': 'application/sparql-results+json',
            'User-Agent': config.get('user_agent', 'DefaultContextRetriever/1.0')
        }
        params = {'query': sparql_query}
        retries = config.get('max_retries', 2)

        logging.debug(f"Querying Wikidata for: {term}")

        for attempt in range(retries + 1):
            try:
                async with self.session.get(endpoint, params=params, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    data = await response.json()
                    logging.debug(f"Wikidata response status: {response.status} for {term}")

                    processed_items = set()

                    for binding in data.get("results", {}).get("bindings", []):
                        item_uri = binding.get("item", {}).get("value")
                        if not item_uri or item_uri in processed_items:
                            continue

                        item_label = binding.get("itemLabel", {}).get("value", term)
                        item_desc = binding.get("itemDescription", {}).get("value")

                        if item_desc:
                            results.append({
                                "source": item_label,
                                "relation": "description",
                                "target": item_desc,
                                "weight": 1.0,
                                "explanation": f"{item_label} is {item_desc}",
                                "source_name": "wikidata",
                                "context_type": "entity",
                                "is_fallback": False,
                            })
                            processed_items.add(item_uri)

                        prop_label = binding.get("propLabel", {}).get("value")
                        value_label = binding.get("valueLabel", {}).get("value")

                        if prop_label and value_label:
                             results.append({
                                "source": item_label,
                                "relation": prop_label,
                                "target": value_label,
                                "weight": 0.9,
                                "explanation": f"{item_label} {prop_label} {value_label}",
                                "source_name": "wikidata",
                                "context_type": "entity",
                                "is_fallback": False,
                            })

                    return results

            except aiohttp.ClientError as e:
                logging.warning(f"Wikidata HTTP Error (Attempt {attempt + 1}/{retries + 1}) for '{term}': {e}")
                if attempt >= retries: raise
            except asyncio.TimeoutError:
                logging.warning(f"Wikidata Timeout (Attempt {attempt + 1}/{retries + 1}) for '{term}'")
                if attempt >= retries: raise
            except json.JSONDecodeError as e:
                 logging.error(f"Wikidata JSON Decode Error for '{term}': {e}")
                 raise

            if attempt < retries:
                 await asyncio.sleep(1)

        return []

    def _process_results(self, results: List[Dict[str, Any]], term: str, base_agent_type: str) -> Dict[str, Any]:
        all_items = results

        def item_priority(item: Dict[str, Any]) -> float:
            weight = item.get("weight", 0.5)
            source_name = item.get("source_name", "")
            is_fallback = item.get("is_fallback", False)

            source_boost = {
                "wikidata": 0.15,
                "conceptnet": 0.05,
            }.get(source_name, 0.0)

            fallback_penalty = 0.5 if is_fallback else 0.0

            term_relevance_bonus = 0.1 if term.lower() in item.get("source","").lower() or term.lower() in item.get("target","").lower() else 0.0

            return weight + source_boost + term_relevance_bonus - fallback_penalty

        all_items.sort(key=item_priority, reverse=True)

        context: Dict[str, Any] = {
             "term": term,
             "agent_type": base_agent_type,
             "items": [],
             "summary": "No relevant context found.",
             "sources_used": [],
             "timestamp": time.time(),
             "has_fallbacks": any(item.get("is_fallback", False) for item in all_items),
             "error": None
        }

        context["items"] = all_items[:self.max_context_items]

        context["summary"] = self._create_optimized_summary(context["items"])

        context["sources_used"] = sorted(list(set(
            item.get("source_name", "unknown") for item in context["items"]
        )))

        logging.info(f"Processed {len(context['items'])} context items for term '{term}'. Summary: {context['summary']}")
        return context

    def _create_optimized_summary(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return "No relevant context found."

        summary_parts = []
        items_used_in_summary = set()

        non_fallback_items = [item for item in items if not item.get("is_fallback", False)]
        items_to_summarize = non_fallback_items if non_fallback_items else items

        for item in items_to_summarize:
            if len(summary_parts) >= 3:
                 break

            fact_tuple = (item.get('source'), item.get('relation'), item.get('target'))
            if fact_tuple in items_used_in_summary:
                 continue

            source_info = f" (via {item['source_name']})" if "source_name" in item else ""
            explanation = item.get('explanation')
            if explanation and len(explanation) < 150:
                 summary_line = f"{explanation}{source_info}"
            else:
                summary_line = f"{item['source']} {item['relation']} {item['target']}{source_info}"

            summary_parts.append(summary_line)
            items_used_in_summary.add(fact_tuple)

        has_fallbacks = any(item.get("is_fallback", False) for item in items)
        if has_fallbacks:
            if non_fallback_items:
                 summary_parts.append("(Note: Some context derived from fallbacks)")
            else:
                 summary_parts.append("(Note: Context based entirely on fallbacks)")

        if summary_parts:
            return ". ".join(summary_parts) + "."
        else:
            return "No significant context found to summarize."

    def _filter_context_for_agent(self, context: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        logging.debug(f"Filtering context for agent '{agent_type}' from base type '{context.get('agent_type')}'")

        relevant_types = set(self.agent_context_types.get(agent_type, ["linguistic", "conceptual", "entity"]))

        filtered_context = {
            "term": context.get("term"),
            "items": [],
            "summary": "",
            "agent_type": agent_type,
            "timestamp": time.time(),
            "original_timestamp": context.get("timestamp"),
            "sources_used": context.get("sources_used", []),
            "has_fallbacks": context.get("has_fallbacks", False),
            "error": context.get("error")
        }

        relevant_items = []
        original_items = context.get("items", [])
        for item in original_items:
            item_type = item.get("context_type")
            if item_type and item_type in relevant_types:
                 relevant_items.append(item)
            elif not item_type:
                 logging.debug(f"Keeping item without context_type during filtering: {item}")
                 relevant_items.append(item)

        filtered_context["items"] = relevant_items[:self.max_context_items]

        filtered_context["summary"] = self._create_optimized_summary(filtered_context["items"])

        logging.info(f"Filtered context for '{agent_type}'. Kept {len(filtered_context['items'])} items out of {len(original_items)}. New summary: {filtered_context['summary']}")

        return filtered_context
    
    async def ensure_session(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logging.error("No active event loop found for context retrieval")
            return False
        
        if not hasattr(self, 'session') or self.session is None or self.session.closed:
            try:
                connector = aiohttp.TCPConnector(
                    limit=5, 
                    ttl_dns_cache=300, 
                    ssl=False
                )
                connector = register_connector(connector)
                
                self.session = register_session(aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    connector=connector,
                    headers={'User-Agent': 'GraphRAG-Agent/1.0'}
                ))
                self.initialized = True
                logging.debug("Created new aiohttp session for context retriever")
                return True
            except Exception as e:
                logging.error(f"Failed to create aiohttp session: {e}")
                return False
        
        return True
    
    async def close_session(self):
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                await self.session.close()
                logging.info("Context retriever session closed")
            except Exception as e:
                logging.warning(f"Error closing context retriever session: {e}")

async def main():
    async with ContextRetriever() as retriever:
        retriever = ContextRetriever(DEFAULT_CONTEXT_CONFIG)
        term = "Barack Obama"
        agent = "aggression_level"

        try:
            print(f"\n--- Request 1: Term='{term}', Agent='{agent}' ---")
            context1 = await retriever.get_context(term, agent)
            print(json.dumps(context1, indent=2))
            print(f"\nMetrics after Request 1:\n{json.dumps(retriever.get_metrics(), indent=2)}")

            print(f"\n--- Request 2: Term='{term}', Agent='explainer' (Testing Cache & Sharing) ---")
            context2 = await retriever.get_context(term, 'explainer')
            print(json.dumps(context2, indent=2))
            print(f"\nMetrics after Request 2:\n{json.dumps(retriever.get_metrics(), indent=2)}")

            print(f"\n--- Request 3: Term='Artificial Intelligence', Agent='{agent}' ---")
            context3 = await retriever.get_context("Narendra Modi", agent)
            print(json.dumps(context3, indent=2))
            print(f"\nMetrics after Request 3:\n{json.dumps(retriever.get_metrics(), indent=2)}")

        except Exception as e:
            logging.error(f"An error occurred in main: {e}", exc_info=True)
        finally:
            await retriever.close_session()
            logging.info("Main finished.")

if __name__ == "__main__":
    asyncio.run(main())