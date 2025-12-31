import sqlite3
import time
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager
from dataclasses import dataclass
import json


@dataclass
class Limit:
    """
    Rappresenta un singolo limite di rate limiting.
    
    Attributes
    ----------
    unit : str
        Unità di misura ('token', 'request')
    value : int
        Numero massimo di unità consentite nel periodo di reset
    reset_unit : str
        Unità di tempo per il reset ('second', 'minute', 'hour', 'day')
    reset_value : int
        Valore numerico per l'unità di tempo (es. 1 per "1 giorno")
    name : str, optional
        Nome identificativo per il limite (default: auto-generato)
    """
    unit: str
    value: int
    reset_unit: str
    reset_value: int
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validazione e generazione automatica del nome."""
        if self.unit not in ['token', 'request']:
            raise ValueError("Unit must be one of: 'token', 'request'")
        
        if self.reset_unit not in ['second', 'minute', 'hour', 'day']:
            raise ValueError("Reset_unit must be one of: 'second', 'minute', 'hour', 'day'")
        
        if self.reset_value <= 0:
            raise ValueError("Reset_value must be a positive integer")
        
        if self.value <= 0:
            raise ValueError("Value must be a positive integer")
        
        # Genera nome automatico se non fornito
        if self.name is None:
            self.name = f"{self.value}_{self.unit}_{self.reset_value}_{self.reset_unit}"
    
    def get_time_window_seconds(self) -> int:
        """Converte l'unità di tempo in secondi."""
        time_windows = {
            'second': 1,
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }
        return time_windows[self.reset_unit] * self.reset_value
    
    def __repr__(self) -> str:
        return f"Limit(name='{self.name}', unit='{self.unit}', value={self.value}, reset_unit='{self.reset_unit}', reset_value={self.reset_value})"


class RateLimiter:
    """
    Rate limiter che usa SQLite per persistenza dei dati.
    
    Supporta multiple limiti simultanei per un singolo utente.
    Ad esempio: 100 richieste al giorno E 10 richieste all'ora.
    
    Supporta diversi tipi di unità di misura:
    - 'token': limite basato sui token
    - 'request': limite basato sulle richieste
    
    E diverse unità di tempo per il reset con valori personalizzabili:
    - 'second': reset ogni N secondi (es. reset_value=5 = ogni 5 secondi)
    - 'minute': reset ogni N minuti (es. reset_value=2 = ogni 2 minuti)
    - 'hour': reset ogni N ore (es. reset_value=3 = ogni 3 ore)
    - 'day': reset ogni N giorni (es. reset_value=1 = ogni giorno)
    """

    def __init__(self, limits: List[Limit], db_path: str = "rate_limiter.db"):
        """
        Inizializza il rate limiter con multiple limiti.
        
        Parameters
        ----------
        limits : List[Limit]
            Lista di limiti da applicare simultaneamente
        db_path : str
            Percorso del database SQLite
        """
        if not limits:
            raise ValueError("At least one limit must be provided")
        
        self._limits = limits
        self._db_path = db_path
        
        # Crea il database e le tabelle se non esistono
        self._init_database()


    def _init_database(self):
        """Inizializza il database SQLite e crea le tabelle necessarie."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id TEXT NOT NULL,
                    limit_name TEXT NOT NULL,
                    window_start INTEGER NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, limit_name, window_start)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_limit_window 
                ON rate_limits(user_id, limit_name, window_start)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_limit_window 
                ON rate_limits(limit_name, window_start)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager per gestire le connessioni SQLite."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row  # Permette accesso per nome colonna
        try:
            yield conn
        finally:
            conn.close()

    def _get_current_window_start(self, limit: Limit) -> int:
        """Calcola l'inizio della finestra temporale corrente per un limite specifico."""
        current_time = int(time.time())
        time_window = limit.get_time_window_seconds()
        return current_time - (current_time % time_window)

    def _cleanup_old_windows(self, conn: sqlite3.Connection):
        """Rimuove le finestre temporali vecchie dal database."""
        current_time = int(time.time())
        for limit in self._limits:
            # Mantieni 2 finestre per ogni limite
            time_window = limit.get_time_window_seconds()
            cutoff_time = current_time - (time_window * 2)
            conn.execute("DELETE FROM rate_limits WHERE limit_name = ? AND window_start < ?", 
                        (limit.name, cutoff_time))

    def update(self, user_id: str, usage: int = 1) -> bool:
        """
        Aggiorna l'uso per un utente specifico per tutti i limiti.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
        usage : int
            Quantità di uso da aggiungere (default: 1)
            
        Returns
        -------
        bool
            True se l'aggiornamento è andato a buon fine
        """
        with self._get_connection() as conn:
            # Pulisci le finestre vecchie
            self._cleanup_old_windows(conn)
            
            # Aggiorna l'uso per ogni limite
            for limit in self._limits:
                window_start = self._get_current_window_start(limit)
                
                # Inserisci o aggiorna l'uso per questa finestra e limite
                conn.execute("""
                    INSERT INTO rate_limits (user_id, limit_name, window_start, usage_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, limit_name, window_start) 
                    DO UPDATE SET usage_count = usage_count + ?
                """, (user_id, limit.name, window_start, usage, usage))
            
            conn.commit()
            return True

    def check(self, user_id: str) -> bool:
        """
        Controlla se un utente ha superato uno qualsiasi dei limiti.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
            
        Returns
        -------
        bool
            True se l'utente può fare richieste, False se ha superato almeno un limite
        """
        for limit in self._limits:
            current_usage = self.get_usage(user_id, limit)
            if current_usage >= limit.value:
                return False
        return True

    def get_usage(self, user_id: str, limit: Optional[Limit] = None) -> int:
        """
        Ottiene l'uso corrente per un utente nella finestra temporale corrente.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
        limit : Limit, optional
            Limite specifico. Se None, restituisce l'uso totale per tutti i limiti.
            
        Returns
        -------
        int
            Numero di richieste nella finestra temporale corrente
        """
        if limit is None:
            # Restituisce l'uso totale per tutti i limiti
            total_usage = 0
            for limit_obj in self._limits:
                total_usage += self.get_usage(user_id, limit_obj)
            return total_usage
        
        window_start = self._get_current_window_start(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT usage_count FROM rate_limits 
                WHERE user_id = ? AND limit_name = ? AND window_start = ?
            """, (user_id, limit.name, window_start))
            
            result = cursor.fetchone()
            return result['usage_count'] if result else 0

    def get_remaining(self, user_id: str, limit: Optional[Limit] = None) -> int:
        """
        Ottiene il numero di richieste rimanenti per un utente.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
        limit : Limit, optional
            Limite specifico. Se None, restituisce il minimo tra tutti i limiti.
            
        Returns
        -------
        int
            Numero di richieste rimanenti
        """
        if limit is None:
            # Restituisce il minimo rimanente tra tutti i limiti
            min_remaining = float('inf')
            for limit_obj in self._limits:
                remaining = self.get_remaining(user_id, limit_obj)
                min_remaining = min(min_remaining, remaining)
            return int(min_remaining) if min_remaining != float('inf') else 0
        
        current_usage = self.get_usage(user_id, limit)
        return max(0, limit.value - current_usage)

    def reset_user(self, user_id: str, limit: Optional[Limit] = None) -> bool:
        """
        Resetta l'uso per un utente specifico.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
        limit : Limit, optional
            Limite specifico. Se None, resetta tutti i limiti per l'utente.
            
        Returns
        -------
        bool
            True se il reset è andato a buon fine
        """
        with self._get_connection() as conn:
            if limit is None:
                conn.execute("DELETE FROM rate_limits WHERE user_id = ?", (user_id,))
            else:
                conn.execute("DELETE FROM rate_limits WHERE user_id = ? AND limit_name = ?", 
                           (user_id, limit.name))
            conn.commit()
            return True

    def get_stats(self, user_id: Optional[str] = None, limit: Optional[Limit] = None) -> Dict[str, Any]:
        """
        Ottiene statistiche di utilizzo.
        
        Parameters
        ----------
        user_id : str, optional
            ID dell'utente specifico. Se None, restituisce statistiche globali.
        limit : Limit, optional
            Limite specifico. Se None, restituisce statistiche per tutti i limiti.
            
        Returns
        -------
        Dict[str, Any]
            Statistiche di utilizzo
        """
        with self._get_connection() as conn:
            if user_id:
                if limit:
                    # Statistiche per utente e limite specifico
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as total_windows,
                            SUM(usage_count) as total_usage,
                            AVG(usage_count) as avg_usage,
                            MAX(usage_count) as max_usage
                        FROM rate_limits 
                        WHERE user_id = ? AND limit_name = ?
                    """, (user_id, limit.name))
                else:
                    # Statistiche per utente specifico (tutti i limiti)
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as total_windows,
                            SUM(usage_count) as total_usage,
                            AVG(usage_count) as avg_usage,
                            MAX(usage_count) as max_usage
                        FROM rate_limits 
                        WHERE user_id = ?
                    """, (user_id,))
            else:
                if limit:
                    # Statistiche globali per limite specifico
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(DISTINCT user_id) as unique_users,
                            COUNT(*) as total_windows,
                            SUM(usage_count) as total_usage,
                            AVG(usage_count) as avg_usage,
                            MAX(usage_count) as max_usage
                        FROM rate_limits
                        WHERE limit_name = ?
                    """, (limit.name,))
                else:
                    # Statistiche globali (tutti i limiti)
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(DISTINCT user_id) as unique_users,
                            COUNT(*) as total_windows,
                            SUM(usage_count) as total_usage,
                            AVG(usage_count) as avg_usage,
                            MAX(usage_count) as max_usage
                        FROM rate_limits
                    """)
            
            result = cursor.fetchone()
            return dict(result) if result else {}

    def cleanup(self) -> int:
        """
        Pulisce le finestre temporali vecchie dal database.
        
        Returns
        -------
        int
            Numero di record rimossi
        """
        with self._get_connection() as conn:
            self._cleanup_old_windows(conn)
            conn.commit()
            return 0  # Il conteggio è gestito in _cleanup_old_windows

    def get_limits(self) -> List[Limit]:
        """
        Restituisce la lista dei limiti configurati.
        
        Returns
        -------
        List[Limit]
            Lista dei limiti
        """
        return self._limits.copy()
    
    def get_limit_by_name(self, name: str) -> Optional[Limit]:
        """
        Restituisce un limite specifico per nome.
        
        Parameters
        ----------
        name : str
            Nome del limite
            
        Returns
        -------
        Limit, optional
            Il limite se trovato, None altrimenti
        """
        for limit in self._limits:
            if limit.name == name:
                return limit
        return None
    
    def get_usage_by_limit(self, user_id: str) -> Dict[str, int]:
        """
        Restituisce l'uso per ogni limite per un utente specifico.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
            
        Returns
        -------
        Dict[str, int]
            Dizionario con nome limite -> uso corrente
        """
        usage_by_limit = {}
        for limit in self._limits:
            usage_by_limit[limit.name] = self.get_usage(user_id, limit)
        return usage_by_limit
    
    def get_remaining_by_limit(self, user_id: str) -> Dict[str, int]:
        """
        Restituisce le richieste rimanenti per ogni limite per un utente specifico.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
            
        Returns
        -------
        Dict[str, int]
            Dizionario con nome limite -> richieste rimanenti
        """
        remaining_by_limit = {}
        for limit in self._limits:
            remaining_by_limit[limit.name] = self.get_remaining(user_id, limit)
        return remaining_by_limit

    def _extract_tokens_from_response(self, response: Union[str, Dict[str, Any]]) -> int:
        """
        Estrae il numero di token dalla risposta del modello/agente.
        
        Parameters
        ----------
        response : Union[str, Dict[str, Any]]
            Risposta del modello/agente (JSON string o dict)
            
        Returns
        -------
        int
            Numero di token estratti, 0 se non trovato
        """
        try:
            # Se è una stringa, prova a parsarla come JSON
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    return 0
            
            # Se è un dict, cerca il campo usage->total_tokens
            if isinstance(response, dict):
                usage = response.get("usage", {})
                if isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens", 0)
                    if isinstance(total_tokens, (int, float)):
                        return int(total_tokens)
            
            return 0
        except Exception:
            return 0

    def update_with_response(self, user_id: str, response: Union[str, Dict[str, Any]]) -> bool:
        """
        Aggiorna il rate limiter basandosi sulla risposta del modello/agente.
        Estrae automaticamente i token se presenti nella risposta.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
        response : Union[str, Dict[str, Any]]
            Risposta del modello/agente (JSON string o dict)
            
        Returns
        -------
        bool
            True se l'aggiornamento è andato a buon fine
        """
        # Estrai i token dalla risposta
        tokens = self._extract_tokens_from_response(response)
        
        # Aggiorna ogni limite basandosi sulla sua unità
        with self._get_connection() as conn:
            # Pulisci le finestre vecchie
            self._cleanup_old_windows(conn)
            
            for limit in self._limits:
                window_start = self._get_current_window_start(limit)
                
                # Determina l'uso basandosi sull'unità del limite
                if limit.unit == "token":
                    usage = tokens
                else:  # request
                    usage = 1
                
                # Inserisci o aggiorna l'uso per questa finestra e limite
                conn.execute("""
                    INSERT INTO rate_limits (user_id, limit_name, window_start, usage_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, limit_name, window_start) 
                    DO UPDATE SET usage_count = usage_count + ?
                """, (user_id, limit.name, window_start, usage, usage))
            
            conn.commit()
            return True

    def check_with_response(self, user_id: str, response: Union[str, Dict[str, Any]]) -> bool:
        """
        Controlla se un utente può fare una richiesta basandosi sulla risposta del modello/agente.
        Estrae automaticamente i token se presenti nella risposta.
        
        Parameters
        ----------
        user_id : str
            ID dell'utente
        response : Union[str, Dict[str, Any]]
            Risposta del modello/agente (JSON string o dict)
            
        Returns
        -------
        bool
            True se l'utente può fare richieste, False se ha superato almeno un limite
        """
        # Estrai i token dalla risposta
        tokens = self._extract_tokens_from_response(response)
        
        # Verifica ogni limite
        for limit in self._limits:
            current_usage = self.get_usage(user_id, limit)
            
            # Determina l'uso aggiuntivo basandosi sull'unità del limite
            if limit.unit == "token":
                additional_usage = tokens
            else:  # request
                additional_usage = 1
            
            # Verifica se l'uso aggiuntivo supererebbe il limite
            if current_usage + additional_usage > limit.value:
                return False
        
        return True

    def __repr__(self) -> str:
        limits_str = ", ".join([str(limit) for limit in self._limits])
        return f"RateLimiter(limits=[{limits_str}], db_path='{self._db_path}')"
    