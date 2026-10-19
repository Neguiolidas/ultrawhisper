# Benchmark: 5 vozes, 2 cenas (2026-10-19)

Áudio: `conversa_5_vozes.mp3` (172s, 43 turnos) — teste do Tabily.
Elenco: ANA, TATI (F0 quase igual, par difícil), RAFA (2 cenas), PAULO, CAROL.

## Resultados

| Abordagem | Precisão | Erros | Notas |
|---|---|---|---|
| Tabily: faster-whisper large-v3 + diarização local | 36/43 (83.7%) | 1 cluster espúrio, 7 turnos | Sem enrollment; clusters anônimos (S1, S2...) |
| Whispero voiceid (enrollment limpo) | 36/43 (83%) | 7 unknowns, 0 trocas | Com enrollment; nomeia falantes |

## Lições

1. **Enrollment limpo importa**: 79% → 83% só removendo contaminação
   (amostra de 10s continha fala de outras pessoas). Sempre extrair
   segmentos do falante alvo, nunca janela fixa.
2. **Erros restantes são fundamentais**: turnos <1s e sobreposições
   derrotam ambas as abordagens. "Unknown" é a resposta certa.
3. **Tradeoff claro**: diarização = sem enrollment, clusters anônimos;
   voiceid = com enrollment, nomes. Complementares, não concorrentes.
4. **ANA × TATI**: ambas as abordagens separaram o par difícil.
5. **RAFA nas 2 cenas**: ambas re-identificaram corretamente.
