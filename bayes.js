
class NaiveBayes{
    /**
     * creates a new classifier.
     * @returns {NaiveBayes}
     */
    NaiveBayes(state) {
        this.state = state;
        this.state = this.state || {
            features: {},
            correlations: {}
        };
    }
    /**
     * trains this classifer with this object.
     * @param {any} the javascript object to train this classifier with.
     * @returns {void}
     */
    train(obj) {
        const _this = this;
        const parameters = Object.keys(obj).map(key => ({
            feature: key,
            attribute: obj[key]
        }));
        parameters.forEach(parameter => _this.insert_feature(parameter));
        parameters.forEach(left => parameters.forEach(right => {
            if (left.feature === right.feature)
                return;
            _this.insert_correlation(left, right);
        }));
    }
    /**
     * classifies this feature with the given object.
     * @param {string} the feature to classify.
     * @param {any} an optional feature
     * @returns {any} the bayes prediction for the given feature.
     */
    classify(feature, obj) {
        const _this = this;
        if (this.state.features[feature] === undefined) {
            return {};
        }
        else if (obj === undefined || Object.keys(obj).length === 0) {
            const sum_1 = Object.keys(this.state.features[feature])
                .map(attribute => _this.state.features[feature][attribute])
                .reduce((acc, count) => acc + count, 0);
            return Object.keys(this.state.features[feature])
                .reduce((acc, attribute) => {
                acc[attribute] = _this.state.features[feature][attribute] / sum_1;
                return acc;
            }, {});
        }
        else {
            const sums_1 = Object.keys(obj).reduce((sums, inner_feature) => {
                sums[inner_feature] = Object.keys(_this.state.correlations[feature]).reduce((sum, attribute) => {
                    if (obj[inner_feature] !== undefined
                        && _this.state.correlations[feature][attribute][inner_feature] !== undefined
                        && _this.state.correlations[feature][attribute][inner_feature][obj[inner_feature]] !== undefined) {
                        return sum + _this.state.correlations[feature][attribute][inner_feature][obj[inner_feature]];
                    }
                    else
                        return sum;
                }, 0);
                return sums;
            }, {});
            const result_1 = Object.keys(this.state.correlations[feature]).reduce((result, attribute) => {
                const probabilities = Object.keys(obj).reduce((probabilities, inner_feature) => {
                    if (obj[inner_feature] !== undefined
                        && _this.state.correlations[feature][attribute][inner_feature] !== undefined
                        && _this.state.correlations[feature][attribute][inner_feature][obj[inner_feature]] !== undefined) {
                        probabilities[inner_feature] = _this.state.correlations[feature][attribute][inner_feature][obj[inner_feature]] / sums_1[inner_feature];
                    }
                    else
                        probabilities[inner_feature] = 0;
                    return probabilities;
                }, {});
                result[attribute] = Object.keys(probabilities).reduce((acc, feature) => acc * probabilities[feature], 1);
                return result;
            }, {});
            const sum_2 = Object.keys(result_1).reduce((acc, attribute) => acc + result_1[attribute], 0);
            return Object.keys(result_1).reduce((acc, attribute) => {
                acc[attribute] = sum_2 > 0 ? result_1[attribute] / sum_2 : 0;
                return acc;
            }, {});
        }
    }
    /**
     * inserts this feature into the feature map, and increments its occurance value +1
     * @param {Parameter} the feature/attribute pair.
     * @returns {void}
     */
    insert_feature(parameter){
        if (this.state.features[parameter.feature] === undefined)
            this.state.features[parameter.feature] = {};
        if (this.state.features[parameter.feature][parameter.attribute] === undefined) {
            this.state.features[parameter.feature][parameter.attribute] = 1;
        }
        else
            this.state.features[parameter.feature][parameter.attribute] += 1;
    }
    /**
     * inserts this correlation in to the correlation map. increments its occurance value +1.
     * This function updates both left and right, feature/attribute pairs, which is a duplication
     * of data, but no more than representing the data in a ND matrix.
     * @param {Parameter} the left feature/attribute pair.
     * @param {Parameter} the right feature/attribute pair.
     * @returns {void}
     */
    insert_correlation(left, right) {
        const _this = this;
        let needs_update = false;
        if (this.state.correlations[left.feature] === undefined)
            this.state.correlations[left.feature] = {};
        if (this.state.correlations[left.feature][left.attribute] === undefined)
            this.state.correlations[left.feature][left.attribute] = {};
        if (this.state.correlations[left.feature][left.attribute][right.feature] === undefined)
            this.state.correlations[left.feature][left.attribute][right.feature] = {};
        if (this.state.correlations[left.feature][left.attribute][right.feature][right.attribute] === undefined) {
            this.state.correlations[left.feature][left.attribute][right.feature][right.attribute] = 1;
            needs_update = true;
        }
        else
            this.state.correlations[left.feature][left.attribute][right.feature][right.attribute] += 1;
        if (needs_update === false)
            return;
        Object.keys(this.state.correlations).forEach(left_feature => {
            Object.keys(_this.state.correlations).forEach(right_feature => {
                if (left_feature === right_feature)
                    return;
                Object.keys(_this.state.correlations[left_feature]).forEach(left_attribute => {
                    Object.keys(_this.state.correlations[right_feature]).forEach(right_attribute => {
                        if (_this.state.correlations[left_feature] === undefined)
                            _this.state.correlations[left_feature] = {};
                        if (_this.state.correlations[left_feature][left_attribute] === undefined)
                            _this.state.correlations[left_feature][left_attribute] = {};
                        if (_this.state.correlations[left_feature][left_attribute][right_feature] === undefined)
                            _this.state.correlations[left_feature][left_attribute][right_feature] = {};
                        if (_this.state.correlations[left_feature][left_attribute][right_feature][right_attribute] === undefined)
                            _this.state.correlations[left_feature][left_attribute][right_feature][right_attribute] = 0;
                    });
                });
            });
        });
    }
}
