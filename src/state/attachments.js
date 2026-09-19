export function appendAttachmentPlaceholders(current, entries) {
  return [...current, ...entries.map(({ id, type, name }) => ({ id, type, name, pending: true }))];
}

export function settleAttachment(current, id, attachment) {
  if (!attachment) return current.filter((item) => item.id !== id);
  return current.map((item) => (item.id === id ? { ...attachment, id } : item));
}
