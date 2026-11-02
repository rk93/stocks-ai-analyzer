// Shared financial primitives. GBX represents pence, GBP represents pounds.
export const currency = (c) =>
  /^[A-Z]{3}$/.test(c) &&
  (() => {
    try {
      new Intl.NumberFormat("en", { style: "currency", currency: c });
      return [
        "GBP",
        "USD",
        "EUR",
        "INR",
        "CAD",
        "AUD",
        "CHF",
        "JPY",
        "HKD",
        "SGD",
        "GBX",
      ].includes(c);
    } catch {
      return false;
    }
  })();
export function validatePosition(r) {
  if (
    !/^[A-Z0-9^][A-Z0-9.^=:/_-]{0,29}$/.test(r.symbol) ||
    !Number.isFinite(r.quantity) ||
    r.quantity <= 0 ||
    !currency(r.currency) ||
    !Number.isFinite(r.cost) ||
    r.cost < 0 ||
    !(r.price == null || (Number.isFinite(r.price) && r.price >= 0))
  )
    throw Error(
      "Check symbol, positive quantity, currency and non-negative unit cost/price",
    );
  return r;
}
