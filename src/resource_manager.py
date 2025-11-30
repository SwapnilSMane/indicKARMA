import gc
import logging
import asyncio
from typing import List, Any
import weakref

logger = logging.getLogger(__name__)

_sessions = []
_connectors = []
_models = []

def register_session(session):
    _sessions.append(weakref.ref(session))
    return session

def register_connector(connector):
    _connectors.append(weakref.ref(connector))
    return connector

def register_model(model):
    _models.append(weakref.ref(model))
    return model

async def cleanup_async_resources():
    for session_ref in _sessions:
        session = session_ref()
        if session is not None and not session.closed:
            try:
                await session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
    
    for connector_ref in _connectors:
        connector = connector_ref()
        if connector is not None and not connector.closed:
            try:
                await connector.close()
            except Exception as e:
                logger.warning(f"Error closing connector: {e}")
    
    try:
        await asyncio.sleep(0.1)
    except:
        pass

def cleanup_models():
    for model_ref in _models:
        model = model_ref()
        if model is not None:
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
            except Exception as e:
                logger.warning(f"Error cleaning up model: {e}")

def cleanup_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error cleaning up GPU memory: {e}")

def cleanup_resources():
    try:
        cleanup_models()
        
        cleanup_gpu_memory()
        
        try:
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                loop.create_task(cleanup_async_resources())
        except RuntimeError:
            pass
        
        gc.collect()
        
        _sessions.clear()
        _connectors.clear()
        _models.clear()
        
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")

def get_resource_stats():
    active_sessions = sum(1 for ref in _sessions if ref() is not None)
    active_connectors = sum(1 for ref in _connectors if ref() is not None)
    active_models = sum(1 for ref in _models if ref() is not None)
    
    return {
        "active_sessions": active_sessions,
        "active_connectors": active_connectors,
        "active_models": active_models,
        "total_registered_sessions": len(_sessions),
        "total_registered_connectors": len(_connectors),
        "total_registered_models": len(_models)
    }