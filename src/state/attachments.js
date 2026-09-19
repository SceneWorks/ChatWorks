export function appendAttachmentPlaceholders(current, entries) {
  return [...current, ...entries.map(({ id, type, name }) => ({ id, type, name, pending: true }))];
}

export function settleAttachment(current, id, attachment) {
  if (!attachment) return current.filter((item) => item.id !== id);
  return current.map((item) => (item.id === id ? { ...attachment, id } : item));
}

// A cancelled placeholder immediately releases its composer slot; late native/browser settlement
// cannot release a second slot or overwrite an attachment in a later conversation.
export function registerPreparation(operations, id, onRelease) {
  const controller = new AbortController();
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    operations.delete(id);
    onRelease();
  };
  const operation = { controller, release, cancel: () => { controller.abort(); release(); } };
  operations.set(id, operation);
  return operation;
}
