def domain_whois(domain:str):

    """
    Get the whois for a given domain:
    
    Args:
        domain (string): the domain
    """

    try:
        from whois import whois
    except ImportError:
        raise ImportError("whois is not installed. Please install it with 'pip install whois'")

    result = whois(domain)
    return str(result)